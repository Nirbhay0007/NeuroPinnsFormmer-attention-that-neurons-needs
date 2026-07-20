# ==============================================================================
# Unified Loss Computation & Optimization Engine for Hodgkin-Huxley PINNs
# ==============================================================================
using Flux
using Zygote
using Statistics

# ------------------------------------------------------------------------------
# 1. Biophysical Constants & Rate Equations
# ------------------------------------------------------------------------------
const C_M   = 1.0f0      # Membrane Capacitance (uF/cm^2)
const G_NA  = 120.0f0    # Max Na+ Conductance (mS/cm^2)
const G_K   = 36.0f0     # Max K+ Conductance (mS/cm^2)
const G_L   = 0.3f0      # Leak Conductance (mS/cm^2)
const E_NA  = 50.0f0     # Na+ Reversal Potential (mV)
const E_K   = -77.0f0    # K+ Reversal Potential (mV)
const E_L   = -54.4f0    # Leak Reversal Potential (mV)

const IC_GROUND_TRUTH = Float32[-65.0, 0.0529, 0.5961, 0.3177]

function alpha_m(V::Real)
    x = -(V + 40.0f0) / 10.0f0
    return iszero(x) ? 1.0f0 : (0.1f0 * (V + 40.0f0)) / (-expm1(x))
end

function beta_m(V::Real)
    return 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)
end

function alpha_h(V::Real)
    return 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)
end

function beta_h(V::Real)
    return 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))
end

function alpha_n(V::Real)
    x = -(V + 55.0f0) / 10.0f0
    return iszero(x) ? 0.1f0 : (0.01f0 * (V + 55.0f0)) / (-expm1(x))
end

function beta_n(V::Real)
    return 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)
end

# ------------------------------------------------------------------------------
# 2. Derivative Extraction Protocol
# ------------------------------------------------------------------------------

"""
    compute_ad_derivatives(model, t_tensor; δt=1f-4)

Computes automatic differentiation derivatives via central finite differences.
Supports input shape (1, 1, B) or pseudo-sequence tensors (1, k, B).
"""
function compute_ad_derivatives(model, t_tensor; δt=1f-4)
    X_pred = model(t_tensor)
    X_plus  = model(t_tensor .+ δt)
    X_minus = model(t_tensor .- δt)
    
    dX_dt = (X_plus .- X_minus) ./ (2.0f0 * δt)
    return X_pred, dX_dt
end

# ------------------------------------------------------------------------------
# 3. Component Residual & Loss Calculations
# ------------------------------------------------------------------------------

"""
    compute_physics_residuals(X_pred, dX_pred_dt, I_ext_value=10.0f0)

Evaluates Hodgkin-Huxley differential equations and computes individual channel MSE losses.
"""
function compute_physics_residuals(X_pred, dX_pred_dt, I_ext_value=10.0f0)
    V = X_pred[1, :, :]
    m = X_pred[2, :, :]
    h = X_pred[3, :, :]
    n = X_pred[4, :, :]

    I_Na = G_NA .* (m .^ 3) .* h .* (V .- E_NA)
    I_K  = G_K  .* (n .^ 4) .* (V .- E_K)
    I_L  = G_L  .* (V .- E_L)

    f_V = (I_ext_value .- I_Na .- I_K .- I_L) ./ C_M
    f_m = alpha_m.(V) .* (1.0f0 .- m) .- beta_m.(V) .* m
    f_h = alpha_h.(V) .* (1.0f0 .- h) .- beta_h.(V) .* h
    f_n = alpha_n.(V) .* (1.0f0 .- n) .- beta_n.(V) .* n

    R_V = dX_pred_dt[1, :, :] .- f_V
    R_m = dX_pred_dt[2, :, :] .- f_m
    R_h = dX_pred_dt[3, :, :] .- f_h
    R_n = dX_pred_dt[4, :, :] .- f_n

    L_RV = mean(abs2, R_V)
    L_Rm = mean(abs2, R_m)
    L_Rh = mean(abs2, R_h)
    L_Rn = mean(abs2, R_n)

    return L_RV, L_Rm, L_Rh, L_Rn
end

"""
    compute_boundary_and_data_losses(model, X_pred, ground_truth_batch)

Evaluates Initial Condition Loss (L_ic) at t = 0.0 ms and Supervised Data Loss (L_data).
"""
function compute_boundary_and_data_losses(model, X_pred, ground_truth_batch)
    t_zero = zeros(Float32, 1, 1, 1)
    X_zero = model(t_zero)[:, 1, 1]
    L_ic = mean(abs2, X_zero .- IC_GROUND_TRUTH)

    L_data = mean(abs2, X_pred .- ground_truth_batch)
    return L_ic, L_data
end

# ------------------------------------------------------------------------------
# 4. NTK State Container & EMA Dynamic Weight Balancer
# ------------------------------------------------------------------------------

mutable struct NTKState
    RV::Float32
    Rm::Float32
    Rh::Float32
    Rn::Float32
    ic::Float32
    data::Float32
end

NTKState() = NTKState(1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0)

"""
    update_ntk_weights_ema!(λ_state, model, batch_t, batch_data, I_ext; α=0.1f0)

Updates adaptive loss multipliers λ_i using Neural Tangent Kernel (NTK) trace sums
smoothed via Exponential Moving Average (EMA).
"""
function update_ntk_weights_ema!(λ_state, model, batch_t, batch_data, I_ext; α=0.1f0)
    _, gs_RV   = Flux.withgradient(m -> compute_physics_residuals(compute_ad_derivatives(m, batch_t)[1], compute_ad_derivatives(m, batch_t)[2], I_ext)[1], model)
    _, gs_Rm   = Flux.withgradient(m -> compute_physics_residuals(compute_ad_derivatives(m, batch_t)[1], compute_ad_derivatives(m, batch_t)[2], I_ext)[2], model)
    _, gs_Rh   = Flux.withgradient(m -> compute_physics_residuals(compute_ad_derivatives(m, batch_t)[1], compute_ad_derivatives(m, batch_t)[2], I_ext)[3], model)
    _, gs_Rn   = Flux.withgradient(m -> compute_physics_residuals(compute_ad_derivatives(m, batch_t)[1], compute_ad_derivatives(m, batch_t)[2], I_ext)[4], model)
    _, gs_ic   = Flux.withgradient(m -> compute_boundary_and_data_losses(m, m(batch_t), batch_data)[1], model)
    _, gs_data = Flux.withgradient(m -> compute_boundary_and_data_losses(m, m(batch_t), batch_data)[2], model)

    function trace_norm(gs)
        if isnothing(gs) || isnothing(gs[1])
            return 1.0f0
        end
        flat_g, _ = Flux.destructure(gs[1])
        return sum(abs2, flat_g) + 1f-8
    end

    tr_RV   = trace_norm(gs_RV)
    tr_Rm   = trace_norm(gs_Rm)
    tr_Rh   = trace_norm(gs_Rh)
    tr_Rn   = trace_norm(gs_Rn)
    tr_ic   = trace_norm(gs_ic)
    tr_data = trace_norm(gs_data)

    mean_target = (tr_RV + tr_Rm + tr_Rh + tr_Rn + tr_ic + tr_data) / 6.0f0

    λ_hat_RV   = mean_target / tr_RV
    λ_hat_Rm   = mean_target / tr_Rm
    λ_hat_Rh   = mean_target / tr_Rh
    λ_hat_Rn   = mean_target / tr_Rn
    λ_hat_ic   = mean_target / tr_ic
    λ_hat_data = mean_target / tr_data

    sum_λ = λ_hat_RV + λ_hat_Rm + λ_hat_Rh + λ_hat_Rn + λ_hat_ic + λ_hat_data
    λ_hat_RV   = 6.0f0 * λ_hat_RV   / sum_λ
    λ_hat_Rm   = 6.0f0 * λ_hat_Rm   / sum_λ
    λ_hat_Rh   = 6.0f0 * λ_hat_Rh   / sum_λ
    λ_hat_Rn   = 6.0f0 * λ_hat_Rn   / sum_λ
    λ_hat_ic   = 6.0f0 * λ_hat_ic   / sum_λ
    λ_hat_data = 6.0f0 * λ_hat_data / sum_λ

    λ_state.RV   = (1.0f0 - α) * λ_state.RV   + α * λ_hat_RV
    λ_state.Rm   = (1.0f0 - α) * λ_state.Rm   + α * λ_hat_Rm
    λ_state.Rh   = (1.0f0 - α) * λ_state.Rh   + α * λ_hat_Rh
    λ_state.Rn   = (1.0f0 - α) * λ_state.Rn   + α * λ_hat_Rn
    λ_state.ic   = (1.0f0 - α) * λ_state.ic   + α * λ_hat_ic
    λ_state.data = (1.0f0 - α) * λ_state.data + α * λ_hat_data

    return λ_state
end

# ------------------------------------------------------------------------------
# 5. Composite Causal & Static Loss Functions
# ------------------------------------------------------------------------------

"""
    total_loss_causal(model, batch_t, batch_data, λ, I_ext; ϵ=1.0f0, use_causality=true)

Computes composite objective function, applying temporal causality mask
across sequence length k if enabled.
"""
function total_loss_causal(model, batch_t, batch_data, λ, I_ext; ϵ=1.0f0, use_causality=true)
    X_pred, dX_dt = compute_ad_derivatives(model, batch_t)
    
    V = X_pred[1, :, :]
    m = X_pred[2, :, :]
    h = X_pred[3, :, :]
    n = X_pred[4, :, :]
    
    I_Na = G_NA .* (m.^3) .* h .* (V .- E_NA)
    I_K  = G_K  .* (n.^4) .* (V .- E_K)
    I_L  = G_L  .* (V .- E_L)
    
    f_V = (I_ext .- I_Na .- I_K .- I_L) ./ C_M
    f_m = alpha_m.(V) .* (1.0f0 .- m) .- beta_m.(V) .* m
    f_h = alpha_h.(V) .* (1.0f0 .- h) .- beta_h.(V) .* h
    f_n = alpha_n.(V) .* (1.0f0 .- n) .- beta_n.(V) .* n
    
    R_V = (dX_dt[1, :, :] .- f_V).^2
    R_m = (dX_dt[2, :, :] .- f_m).^2
    R_h = (dX_dt[3, :, :] .- f_h).^2
    R_n = (dX_dt[4, :, :] .- f_n).^2
    
    L_steps = vec(mean(λ.RV .* R_V .+ λ.Rm .* R_m .+ λ.Rh .* R_h .+ λ.Rn .* R_n, dims=2))
    
    if use_causality && length(L_steps) > 1
        cum_losses = cumsum(L_steps)
        accumulated_losses = cat(0.0f0, cum_losses[1:end-1]; dims=1)
        w = exp.(-ϵ .* accumulated_losses)
        L_phys_total = sum(w .* L_steps)
    else
        L_phys_total = mean(L_steps)
    end
    
    L_ic, L_data = compute_boundary_and_data_losses(model, X_pred, batch_data)
    
    return L_phys_total + (λ.ic * L_ic) + (λ.data * L_data)
end

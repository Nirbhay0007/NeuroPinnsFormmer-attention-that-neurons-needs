# ==============================================================================
# Model Architectures Suite for Hodgkin-Huxley PINNsFormer & Baselines
# ==============================================================================
using Flux
using NNlib
using LinearAlgebra

# ------------------------------------------------------------------------------
# 1. Custom & Benchmark Activation Layers
# ------------------------------------------------------------------------------

"""
    WaveletActivation(d_model::Int)

Localized Mexican Hat / Dual-frequency Wavelet activation mapping:
    f(x) = w1 * sin(x) + w2 * cos(x)
with learnable scale parameters w1, w2.
"""
struct WaveletActivation
    w1::AbstractVector{Float32}
    w2::AbstractVector{Float32}
end
Flux.@layer WaveletActivation

WaveletActivation(d_model::Int) = WaveletActivation(ones(Float32, d_model), ones(Float32, d_model))
(w::WaveletActivation)(x) = (w.w1 .* sin.(x)) .+ (w.w2 .* cos.(x))

"""
Helper function to construct activation functions for layer building.
"""
function build_activation(act_sym::Symbol, d_model::Int)
    if act_sym == :wavelet
        return WaveletActivation(d_model)
    elseif act_sym == :tanh
        return x -> tanh.(x)
    elseif act_sym == :gelu
        return x -> NNlib.gelu.(x)
    elseif act_sym == :siren
        return x -> sin.(30.0f0 .* x)
    elseif act_sym == :silu
        return x -> NNlib.silu.(x)
    else
        error("Unsupported activation symbol: $act_sym")
    end
end

# ------------------------------------------------------------------------------
# 2. Transformer Building Blocks
# ------------------------------------------------------------------------------

struct TransformerBlock
    mha::Flux.MultiHeadAttention
    norm1::Flux.LayerNorm
    ffn::Flux.Chain
    norm2::Flux.LayerNorm
end
Flux.@layer TransformerBlock

function TransformerBlock(d_model::Int, n_heads::Int, act_fn)
    return TransformerBlock(
        Flux.MultiHeadAttention(d_model, nheads = n_heads),
        Flux.LayerNorm(d_model),
        Flux.Chain(Flux.Dense(d_model => d_model), act_fn),
        Flux.LayerNorm(d_model)
    )
end

function (t::TransformerBlock)(x)
    attn_out = t.mha(x, x, x)[1][1][1]
    x = t.norm1(x .+ attn_out)
    ffn_out = t.ffn(x)
    x = t.norm2(x .+ ffn_out)
    return x
end

# ------------------------------------------------------------------------------
# 3. Generalized PINNsFormer Model Architecture
# ------------------------------------------------------------------------------

"""
    PINNsFormer

Generalized Transformer-based Physics-Informed Neural Network supporting:
- Configurable activation function (`:wavelet`, `:tanh`, `:gelu`, `:siren`)
- Configurable sequence look-ahead horizon `k` (e.g., 4, 8, 10, 12, 16)
- Configurable model depth (number of Transformer encoder blocks)
- Configurable embedding dimension `d_model` and attention heads `n_heads`
"""
struct PINNsFormer
    fine_proj::Flux.Dense
    coarse_proj::Flux.Dense
    pe::Array{Float32, 3}
    wavelet::Any
    fine_encoders::Vector{TransformerBlock}
    coarse_encoders::Vector{TransformerBlock}
    M_align::Array{Float32, 4}
    head::Flux.Dense
    act_type::Symbol
end
Flux.@layer PINNsFormer

function PINNsFormer(; d_model::Int=32, k::Int=10, n_heads::Int=4, depth::Int=1, act_type::Symbol=:wavelet, dt_fine=0.05f0, dt_coarse=0.5f0)
    pe = zeros(Float32, d_model, k)
    for j in 1:k
        for i in 0:(div(d_model, 2) - 1)
            denom = 10000.0f0^(2f0 * i / d_model)
            pe[2i+1, j] = sin((j - 1) / denom)
            pe[2i+2, j] = cos((j - 1) / denom)
        end
    end
    pe_3d = reshape(pe, d_model, k, 1)
    
    M = zeros(Float32, k, k)
    for i in 1:k
        t_target = (i - 1) * dt_fine
        val = (t_target / dt_coarse) + 1.0f0
        low_idx = clamp(floor(Int, val), 1, k)
        high_idx = clamp(ceil(Int, val), 1, k)
        if low_idx == high_idx
            M[i, low_idx] = 1.0f0
        else
            rem_val = val - low_idx
            M[i, low_idx] = 1.0f0 - rem_val
            M[i, high_idx] = rem_val
        end
    end
    M_align_4d = reshape(M, 1, k, k, 1)

    act_layer = build_activation(act_type, d_model)
    
    fine_encs = [TransformerBlock(d_model, n_heads, build_activation(act_type, d_model)) for _ in 1:depth]
    coarse_encs = [TransformerBlock(d_model, n_heads, build_activation(act_type, d_model)) for _ in 1:depth]

    return PINNsFormer(
        Flux.Dense(1 => d_model),
        Flux.Dense(1 => d_model),
        pe_3d,
        act_layer,
        fine_encs,
        coarse_encs,
        M_align_4d,
        Flux.Dense(d_model => 4),
        act_type
    )
end

function generate_trajectories(t_batch, k; dt_fine=0.05f0, dt_coarse=0.5f0)
    steps = reshape(0:(k-1), 1, k, 1)
    return t_batch .+ (steps .* dt_fine), t_batch .+ (steps .* dt_coarse)
end

function align_coarse_path(H_coarse::AbstractArray, M_align::AbstractArray)
    d_model, k, B = size(H_coarse)
    return dropdims(sum(reshape(H_coarse, d_model, 1, k, B) .* M_align, dims=3), dims=3)
end

function (m::PINNsFormer)(t_batch)
    k = size(m.pe, 2)
    T_fine, T_coarse = generate_trajectories(t_batch, k)
    X_fine = m.fine_proj(T_fine) .+ m.pe
    X_coarse = m.coarse_proj(T_coarse) .+ m.pe
    
    Z_fine = m.wavelet(X_fine)
    Z_coarse = m.wavelet(X_coarse)
    
    H_fine = Z_fine
    H_coarse = Z_coarse
    
    for block in m.fine_encoders
        H_fine = block(H_fine)
    end
    for block in m.coarse_encoders
        H_coarse = block(H_coarse)
    end
    
    H_coarse_aligned = align_coarse_path(H_coarse, m.M_align)
    H_unified = H_fine .+ H_coarse_aligned
    raw_out = m.head(H_unified)
    
    V = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :, :])
    gating = NNlib.sigmoid.(raw_out[2:4, :, :])
    return cat(V, gating; dims=1)
end

# ------------------------------------------------------------------------------
# 4. External SciML Baseline Architectures
# ------------------------------------------------------------------------------

"""
    VanillaMLP

Standard Multilayer Perceptron PINN mapping time t -> [V, m, h, n].
Uses continuous global activations (tanh or GELU).
"""
struct VanillaMLP
    net::Flux.Chain
end
Flux.@layer VanillaMLP

function VanillaMLP(; hidden_dim::Int=64, num_layers::Int=4, act_fn=tanh)
    layers = []
    push!(layers, Flux.Dense(1 => hidden_dim, act_fn))
    for _ in 1:(num_layers - 2)
        push!(layers, Flux.Dense(hidden_dim => hidden_dim, act_fn))
    end
    push!(layers, Flux.Dense(hidden_dim => 4))
    return VanillaMLP(Flux.Chain(layers...))
end

function (m::VanillaMLP)(t_batch)
    in_dims = ndims(t_batch)
    if in_dims == 3
        B = size(t_batch, 3)
        t_flat = reshape(t_batch, 1, B)
        raw_out = m.net(t_flat)
        V = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :])
        gating = NNlib.sigmoid.(raw_out[2:4, :])
        out_flat = cat(V, gating; dims=1)
        return reshape(out_flat, 4, 1, B)
    else
        raw_out = m.net(t_batch)
        V = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :])
        gating = NNlib.sigmoid.(raw_out[2:4, :])
        return cat(V, gating; dims=1)
    end
end

"""
    SIREN_PINN

Sinusoidal Representation Network (SIREN) for 1D time mapping to HH states.
Uses periodic sin(omega0 * (W*x + b)) activation functions.
"""
struct SIREN_PINN
    net::Flux.Chain
end
Flux.@layer SIREN_PINN

function SIREN_PINN(; hidden_dim::Int=64, num_layers::Int=4, omega0::Float32=30.0f0)
    siren_act = x -> sin.(omega0 .* x)
    layers = []
    push!(layers, Flux.Dense(1 => hidden_dim, siren_act))
    for _ in 1:(num_layers - 2)
        push!(layers, Flux.Dense(hidden_dim => hidden_dim, siren_act))
    end
    push!(layers, Flux.Dense(hidden_dim => 4))
    return SIREN_PINN(Flux.Chain(layers...))
end

function (m::SIREN_PINN)(t_batch)
    in_dims = ndims(t_batch)
    if in_dims == 3
        B = size(t_batch, 3)
        t_flat = reshape(t_batch, 1, B)
        raw_out = m.net(t_flat)
        V = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :])
        gating = NNlib.sigmoid.(raw_out[2:4, :])
        out_flat = cat(V, gating; dims=1)
        return reshape(out_flat, 4, 1, B)
    else
        raw_out = m.net(t_batch)
        V = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :])
        gating = NNlib.sigmoid.(raw_out[2:4, :])
        return cat(V, gating; dims=1)
    end
end

"""
    ModifiedMLP_PINN (Wang et al. Architecture)

Forward-connected gated neural network designed by Wang et al. (2021) to handle
gradient pathologies in physics-informed neural networks:
    U = tanh(W_u * t + b_u)
    V = tanh(W_v * t + b_v)
    H^{(1)} = tanh(W_1 * t + b_1)
    H^{(l+1)} = (1 - Z^{(l)}) .* U + Z^{(l)} .* V, where Z^{(l)} = tanh(W_l * H^{(l)} + b_l)
"""
struct ModifiedMLP_PINN
    u_dense::Flux.Dense
    v_dense::Flux.Dense
    in_dense::Flux.Dense
    hidden_denses::Vector{Flux.Dense}
    out_dense::Flux.Dense
end
Flux.@layer ModifiedMLP_PINN

function ModifiedMLP_PINN(; hidden_dim::Int=64, num_layers::Int=4)
    u_dense = Flux.Dense(1 => hidden_dim, tanh)
    v_dense = Flux.Dense(1 => hidden_dim, tanh)
    in_dense = Flux.Dense(1 => hidden_dim, tanh)
    hidden_denses = [Flux.Dense(hidden_dim => hidden_dim, tanh) for _ in 1:(num_layers - 2)]
    out_dense = Flux.Dense(hidden_dim => 4)
    return ModifiedMLP_PINN(u_dense, v_dense, in_dense, hidden_denses, out_dense)
end

function (m::ModifiedMLP_PINN)(t_batch)
    in_dims = ndims(t_batch)
    t_flat = in_dims == 3 ? reshape(t_batch, 1, size(t_batch, 3)) : t_batch
    
    U = m.u_dense(t_flat)
    V = m.v_dense(t_flat)
    H = m.in_dense(t_flat)
    
    for dense in m.hidden_denses
        Z = dense(H)
        H = (1.0f0 .- Z) .* U .+ Z .* V
    end
    
    raw_out = m.out_dense(H)
    V_pred = -20.0f0 .+ 70.0f0 .* tanh.(raw_out[1:1, :])
    gating = NNlib.sigmoid.(raw_out[2:4, :])
    out_flat = cat(V_pred, gating; dims=1)
    
    return in_dims == 3 ? reshape(out_flat, 4, 1, size(t_batch, 3)) : out_flat
end

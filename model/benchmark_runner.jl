# ==============================================================================
# Automated Comparative Benchmark Driver & Hardware Metric Logger
# ==============================================================================
using CSV
using DataFrames
using Flux
using Optim
using Random
using Statistics
using Dates
using JSON

include("architectures.jl")
include("loss_and_optimization.jl")

"""
    prepare_dataloader(HH_data::DataFrame, k::Int=10; batch_size::Int=64)

Constructs pseudo-sequence target tensors of shape (4, k, max_idx) for sequence horizon k.
"""
function prepare_dataloader(HH_data::DataFrame, k::Int=10; batch_size::Int=64)
    stride = 5
    max_idx = size(HH_data, 1) - (k - 1) * stride
    if max_idx <= 0
        error("Sequence length k=$k exceeds dataset length $(size(HH_data, 1))")
    end

    t_train = reshape(Float32.(HH_data.t[1:max_idx]), 1, 1, max_idx)
    gt_tensor = zeros(Float32, 4, k, max_idx)
    for s in 1:max_idx
        for γ in 1:k
            row_idx = s + (γ - 1) * stride
            gt_tensor[1, γ, s] = Float32(HH_data.V[row_idx])
            gt_tensor[2, γ, s] = Float32(HH_data.m[row_idx])
            gt_tensor[3, γ, s] = Float32(HH_data.h[row_idx])
            gt_tensor[4, γ, s] = Float32(HH_data.n[row_idx])
        end
    end

    return Flux.DataLoader((t_train, gt_tensor), batchsize=batch_size, shuffle=true), t_train, gt_tensor
end

"""
    count_parameters(model)

Calculates the total number of trainable floating-point parameters in a Flux model.
"""
function count_parameters(model)
    flat_params, _ = Flux.destructure(model)
    return length(flat_params)
end

"""
    evaluate_relative_l2_error(model, HH_data)

Computes relative L2 state vector error and component MSE against ground-truth data.
"""
function evaluate_relative_l2_error(model, HH_data)
    t_full_tensor = reshape(Float32.(HH_data.t), 1, 1, length(HH_data.t))
    full_prediction = model(t_full_tensor)
    
    V_pred = full_prediction[1, 1, :]
    m_pred = full_prediction[2, 1, :]
    h_pred = full_prediction[3, 1, :]
    n_pred = full_prediction[4, 1, :]
    
    V_gt = Float32.(HH_data.V)
    m_gt = Float32.(HH_data.m)
    h_gt = Float32.(HH_data.h)
    n_gt = Float32.(HH_data.n)
    
    mse_V = mean(abs2, V_pred .- V_gt)
    mse_m = mean(abs2, m_pred .- m_gt)
    mse_h = mean(abs2, h_pred .- h_gt)
    mse_n = mean(abs2, n_pred .- n_gt)
    
    l2_V = norm(V_pred .- V_gt) / (norm(V_gt) + 1e-8)
    l2_m = norm(m_pred .- m_gt) / (norm(m_gt) + 1e-8)
    l2_h = norm(h_pred .- h_gt) / (norm(h_gt) + 1e-8)
    l2_n = norm(n_pred .- n_gt) / (norm(n_gt) + 1e-8)
    
    l2_total = sqrt(l2_V^2 + l2_m^2 + l2_h^2 + l2_n^2) / 2.0f0
    
    return (
        l2_total = l2_total,
        l2_V = l2_V, l2_m = l2_m, l2_h = l2_h, l2_n = l2_n,
        mse_V = mse_V, mse_m = mse_m, mse_h = mse_h, mse_n = mse_n,
        V_pred = V_pred, m_pred = m_pred, h_pred = h_pred, n_pred = n_pred
    )
end

"""
    instantiate_model(model_id::String, k::Int)

Factory function returning the specified model architecture instance.
"""
function instantiate_model(model_id::String, k::Int)
    if model_id == "M1"  # Ours (Base Model)
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M2"  # tanh Activation Variant
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:tanh)
    elseif model_id == "M3"  # GELU Activation Variant
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:gelu)
    elseif model_id == "M4"  # SIREN Spectral Activation Variant
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:siren)
    elseif model_id == "M5"  # Static Loss Weight Control (-NTK)
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M6"  # Context Horizon Variant (k passed explicitly)
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M7"  # Capacity Variant (Reduced depth & heads)
        return PINNsFormer(; d_model=16, k=k, n_heads=2, depth=1, act_type=:wavelet)
    elseif model_id == "M-MLP"  # External Baseline: Vanilla MLP
        return VanillaMLP(; hidden_dim=64, num_layers=4, act_fn=tanh)
    elseif model_id == "M-SIREN" # External Baseline: SIREN Network
        return SIREN_PINN(; hidden_dim=64, num_layers=4, omega0=30.0f0)
    elseif model_id == "M-Wang"  # External Baseline: Modified Gated MLP
        return ModifiedMLP_PINN(; hidden_dim=64, num_layers=4)
    else
        error("Unknown model ID: $model_id")
    end
end

"""
    train_model_single_run(model, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)

Executes dual-stage optimization (Adam + L-BFGS) for a single initialization seed.
"""
function train_model_single_run(model, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
    I_ext = 10.0f0
    λ_state = NTKState()
    opt_adam = Flux.setup(Flux.Adam(1e-3), model)
    
    t_start = time()
    
    # Stage 1: Adam
    for epoch in 1:adam_epochs
        for (batch_t, batch_data) in dataloader
            if use_ntk
                update_ntk_weights_ema!(λ_state, model, batch_t, batch_data, I_ext; α=0.1f0)
            end
            
            loss_val, grads = Flux.withgradient(
                m -> total_loss_causal(m, batch_t, batch_data, λ_state, I_ext; use_causality=true), 
                model
            )
            Flux.update!(opt_adam, model, grads[1])
        end
    end
    
    # Stage 2: L-BFGS
    flat_params, re = Flux.destructure(model)
    sample_t, sample_data = first(dataloader)

    function lbfgs_obj(p)
        return total_loss_causal(re(p), sample_t, sample_data, λ_state, I_ext; use_causality=true)
    end

    function lbfgs_grad!(g, p)
        _, grads = Flux.withgradient(
            m -> total_loss_causal(m, sample_t, sample_data, λ_state, I_ext; use_causality=true), 
            re(p)
        )
        flat_grads, _ = Flux.destructure(grads[1])
        g .= flat_grads
    end

    res = Optim.optimize(
        lbfgs_obj, 
        lbfgs_grad!, 
        flat_params, 
        LBFGS(m=10), 
        Optim.Options(iterations=lbfgs_epochs, show_trace=false)
    )
    
    trained_model = re(Optim.minimizer(res))
    elapsed_time = time() - t_start
    
    return trained_model, elapsed_time
end

"""
    run_benchmark_suite(; adam_epochs=500, lbfgs_epochs=100, num_seeds=3)

Executes complete model taxonomy benchmark across multiple seeds and logs results to JSON.
"""
function run_benchmark_suite(; adam_epochs=500, lbfgs_epochs=100, num_seeds=3)
    data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
    if !isfile(data_path)
        error("Dataset file not found at $data_path")
    end
    HH_data = CSV.read(data_path, DataFrame)

    model_ids = ["M1", "M2", "M3", "M4", "M5", "M7", "M-MLP", "M-SIREN", "M-Wang"]
    results_dict = Dict{String, Any}()

    println("=========================================================================")
    println("STARTING SYSTEMATIC SciML BENCHMARK SUITE")
    println("Adam Epochs: $adam_epochs | L-BFGS Epochs: $lbfgs_epochs | Seeds: $num_seeds")
    println("=========================================================================\n")

    for id in model_ids
        println("--> Benchmarking Model: [$id] across $num_seeds seeds...")
        k_val = id == "M6" ? 8 : 10
        dataloader, _, _ = prepare_dataloader(HH_data, k_val; batch_size=64)
        use_ntk = id != "M5" # Disable NTK dynamic balancing for M5
        
        seed_l2_errors = Float32[]
        seed_runtimes = Float64[]
        param_count = 0
        best_model_run = nothing
        best_l2 = Inf32

        for seed in 1:num_seeds
            Random.seed!(42 + seed * 100)
            model_inst = instantiate_model(id, k_val)
            param_count = count_parameters(model_inst)
            
            trained_m, wall_time = train_model_single_run(
                model_inst, dataloader; 
                adam_epochs=adam_epochs, 
                lbfgs_epochs=lbfgs_epochs, 
                use_ntk=use_ntk
            )
            
            eval_metrics = evaluate_relative_l2_error(trained_m, HH_data)
            push!(seed_l2_errors, eval_metrics.l2_total)
            push!(seed_runtimes, wall_time)

            if eval_metrics.l2_total < best_l2
                best_l2 = eval_metrics.l2_total
                best_model_run = (model = trained_m, metrics = eval_metrics)
            end
        end

        mean_l2 = mean(seed_l2_errors)
        std_l2 = std(seed_l2_errors)
        mean_time = mean(seed_runtimes)

        println("    Params: $param_count | Mean Rel L2 Error: $(round(mean_l2, digits=5)) ± $(round(std_l2, digits=5)) | Mean Time: $(round(mean_time, digits=2))s")

        results_dict[id] = Dict(
            "model_id" => id,
            "param_count" => param_count,
            "mean_l2_error" => mean_l2,
            "std_l2_error" => std_l2,
            "mean_runtime_sec" => mean_time,
            "seed_l2_errors" => seed_l2_errors,
            "best_l2_error" => best_l2,
            "best_predictions" => Dict(
                "V" => Vector{Float64}(best_model_run.metrics.V_pred),
                "m" => Vector{Float64}(best_model_run.metrics.m_pred),
                "h" => Vector{Float64}(best_model_run.metrics.h_pred),
                "n" => Vector{Float64}(best_model_run.metrics.n_pred)
            )
        )
    end

    # Save benchmark outputs to JSON
    output_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports"
    mkpath(output_dir)
    json_path = joinpath(output_dir, "benchmark_results.json")
    open(json_path, "w") do io
        JSON.print(io, results_dict, 4)
    end
    println("\n[SUCCESS] Benchmark output saved cleanly to $json_path")
    return results_dict
end

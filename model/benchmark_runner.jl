# ==============================================================================
# Production Benchmark Driver, Checkpoint Serialization & Weight Injection Suite
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

# Fixed Single Deterministic Seed for 100% Reproducibility & Fast Single-Run Training
const BENCHMARK_SEEDS = [42]

"""
    prepare_dataloader(HH_data::DataFrame, k::Int=10; batch_size::Int=64)
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
"""
function count_parameters(model)
    flat_params, _ = Flux.destructure(model)
    return length(flat_params)
end

"""
    evaluate_relative_l2_error(model, HH_data)
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
    save_checkpoint_weights(save_path::String, model_id::String, seed::Int, flat_params::Vector{Float32}, metrics, loss_history)
"""
function save_checkpoint_weights(save_path::String, model_id::String, seed::Int, flat_params::Vector{Float32}, metrics, loss_history)
    mkpath(dirname(save_path))
    checkpoint_data = Dict(
        "model_id" => model_id,
        "seed" => seed,
        "param_count" => length(flat_params),
        "weights" => Vector{Float64}(flat_params),
        "metrics" => Dict(
            "l2_total" => Float64(metrics.l2_total),
            "l2_V" => Float64(metrics.l2_V),
            "l2_m" => Float64(metrics.l2_m),
            "l2_h" => Float64(metrics.l2_h),
            "l2_n" => Float64(metrics.l2_n),
            "mse_V" => Float64(metrics.mse_V),
            "mse_m" => Float64(metrics.mse_m),
            "mse_h" => Float64(metrics.mse_h),
            "mse_n" => Float64(metrics.mse_n)
        ),
        "predictions" => Dict(
            "V" => Vector{Float64}(metrics.V_pred),
            "m" => Vector{Float64}(metrics.m_pred),
            "h" => Vector{Float64}(metrics.h_pred),
            "n" => Vector{Float64}(metrics.n_pred)
        ),
        "loss_history" => Vector{Float64}(loss_history)
    )

    open(save_path, "w") do io
        JSON.print(io, checkpoint_data, 4)
    end
end

"""
    save_model_checkpoints(model_id::String, seed::Int, final_model, best_weights, milestone_snapshots, history, final_metrics, HH_data, checkpoint_dir::String)
"""
function save_model_checkpoints(model_id::String, seed::Int, final_model, best_weights::Vector{Float32}, milestone_snapshots::Dict{Int, Vector{Float32}}, history, final_metrics, HH_data, checkpoint_dir::String)
    model_save_dir = joinpath(checkpoint_dir, model_id, "seed_$seed")
    mkpath(model_save_dir)

    # 1. Save Final Model Checkpoint (End of training)
    final_params, _ = Flux.destructure(final_model)
    save_checkpoint_weights(joinpath(model_save_dir, "checkpoint_final.json"), model_id, seed, final_params, final_metrics, history.total)

    # 2. Save Best Model Checkpoint (Lowest loss epoch)
    _, re = Flux.destructure(final_model)
    best_model_inst = re(best_weights)
    best_metrics = evaluate_relative_l2_error(best_model_inst, HH_data)
    save_checkpoint_weights(joinpath(model_save_dir, "checkpoint_best.json"), model_id, seed, best_weights, best_metrics, history.total)

    # 3. Save Significant Milestone Epoch Checkpoints
    milestone_dir = joinpath(model_save_dir, "milestones")
    for (ep, w_vec) in milestone_snapshots
        m_inst = re(w_vec)
        m_eval = evaluate_relative_l2_error(m_inst, HH_data)
        save_checkpoint_weights(joinpath(milestone_dir, "checkpoint_epoch_$(ep).json"), model_id, seed, w_vec, m_eval, history.total[1:min(ep, length(history.total))])
    end

    println("    [CHECKPOINT STORED] $model_id (Seed $seed): Saved final, best, and $(length(milestone_snapshots)) milestone checkpoints.")
end

"""
    load_model_checkpoint(model_id::String, seed::Int=42, checkpoint_type::String="best"; k_val::Int=10, checkpoint_dir::String="")
"""
function load_model_checkpoint(model_id::String, seed::Int=42; checkpoint_type::String="best", k_val::Int=10, checkpoint_dir::String="")
    if isempty(checkpoint_dir)
        checkpoint_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports\checkpoints"
    end
    
    filename = startswith(checkpoint_type, "epoch_") ? joinpath("milestones", "checkpoint_$(checkpoint_type).json") : "checkpoint_$(checkpoint_type).json"
    json_path = joinpath(checkpoint_dir, model_id, "seed_$seed", filename)
    if !isfile(json_path)
        error("Checkpoint file not found: $json_path")
    end

    data = JSON.parsefile(json_path)
    weights = Float32.(data["weights"])

    model_shell = instantiate_model(model_id, k_val)
    _, re = Flux.destructure(model_shell)
    restored_model = re(weights)
    
    println("[WEIGHT INJECTION SUCCESS] Restored [$model_id | Seed $seed | Type: $checkpoint_type] from $json_path")
    return restored_model, data
end

"""
    instantiate_model(model_id::String, k::Int)
"""
function instantiate_model(model_id::String, k::Int)
    if model_id == "M1"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M2"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:tanh)
    elseif model_id == "M3"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:gelu)
    elseif model_id == "M4"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:siren)
    elseif model_id == "M5"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M6"
        return PINNsFormer(; d_model=32, k=k, n_heads=4, depth=1, act_type=:wavelet)
    elseif model_id == "M7"
        return PINNsFormer(; d_model=16, k=k, n_heads=2, depth=1, act_type=:wavelet)
    elseif model_id == "M-MLP"
        return VanillaMLP(; hidden_dim=64, num_layers=4, act_fn=tanh)
    elseif model_id == "M-SIREN"
        return SIREN_PINN(; hidden_dim=64, num_layers=4, omega0=30.0f0)
    elseif model_id == "M-Wang"
        return ModifiedMLP_PINN(; hidden_dim=64, num_layers=4)
    else
        error("Unknown model ID: $model_id")
    end
end

"""
    train_model_single_run(model, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true, seed=42)
"""
function train_model_single_run(model, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true, seed=42)
    Random.seed!(seed)
    I_ext = 10.0f0
    λ_state = NTKState()
    opt_adam = Flux.setup(Flux.Adam(1e-3), model)
    loss_history = Float32[]
    
    best_loss = Inf32
    best_weights, _ = Flux.destructure(model)
    milestone_snapshots = Dict{Int, Vector{Float32}}()

    milestone_epochs = [
        max(1, div(adam_epochs, 10)),
        div(adam_epochs, 4),
        div(adam_epochs, 2),
        div(3 * adam_epochs, 4),
        adam_epochs
    ]

    t_start = time()
    
    # Stage 1: Adam
    for epoch in 1:adam_epochs
        total_epoch_loss = 0.0f0
        for (batch_t, batch_data) in dataloader
            if use_ntk
                update_ntk_weights_ema!(λ_state, model, batch_t, batch_data, I_ext; α=0.1f0)
            end
            
            loss_val, grads = Flux.withgradient(
                m -> total_loss_causal(m, batch_t, batch_data, λ_state, I_ext; use_causality=true), 
                model
            )
            Flux.update!(opt_adam, model, grads[1])
            total_epoch_loss += loss_val
        end
        push!(loss_history, total_epoch_loss)

        if total_epoch_loss < best_loss
            best_loss = total_epoch_loss
            w_flat, _ = Flux.destructure(model)
            best_weights = copy(w_flat)
        end

        if epoch in milestone_epochs
            w_flat, _ = Flux.destructure(model)
            milestone_snapshots[epoch] = copy(w_flat)
        end
    end
    
    # Stage 2: L-BFGS
    flat_params, re = Flux.destructure(model)
    sample_t, sample_data = first(dataloader)

    function lbfgs_obj(p)
        l = total_loss_causal(re(p), sample_t, sample_data, λ_state, I_ext; use_causality=true)
        push!(loss_history, l)
        
        if l < best_loss
            best_loss = l
            best_weights .= p
        end
        return l
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
    
    history = (total = loss_history,)
    return trained_model, elapsed_time, history, best_weights, milestone_snapshots
end

"""
    run_benchmark_suite(; adam_epochs=1000, lbfgs_epochs=200, seeds=BENCHMARK_SEEDS)
"""
function run_benchmark_suite(; adam_epochs=1000, lbfgs_epochs=200, seeds=BENCHMARK_SEEDS)
    data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
    HH_data = CSV.read(data_path, DataFrame)

    model_ids = ["M1", "M2", "M3", "M4", "M5", "M7", "M-MLP", "M-SIREN", "M-Wang"]
    results_dict = Dict{String, Any}()
    checkpoint_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports\checkpoints"

    println("=========================================================================")
    println("REPRODUCIBLE SINGLE-SEED BENCHMARK & CHECKPOINT SERIALIZATION SUITE")
    println("Deterministic Seed: $seeds")
    println("=========================================================================\n")

    for id in model_ids
        println("--> Benchmarking Model: [$id] with seed $seeds...")
        k_val = id == "M6" ? 8 : 10
        dataloader, _, _ = prepare_dataloader(HH_data, k_val; batch_size=64)
        use_ntk = id != "M5"
        
        seed_l2_errors = Float32[]
        seed_runtimes = Float64[]
        param_count = 0
        best_model_run = nothing
        best_l2 = Inf32

        for seed in seeds
            Random.seed!(seed)
            model_inst = instantiate_model(id, k_val)
            param_count = count_parameters(model_inst)
            
            trained_m, wall_time, history, best_w, milestones = train_model_single_run(
                model_inst, dataloader; 
                adam_epochs=adam_epochs, 
                lbfgs_epochs=lbfgs_epochs, 
                use_ntk=use_ntk,
                seed=seed
            )
            
            eval_metrics = evaluate_relative_l2_error(trained_m, HH_data)
            push!(seed_l2_errors, eval_metrics.l2_total)
            push!(seed_runtimes, wall_time)

            save_model_checkpoints(id, seed, trained_m, best_w, milestones, history, eval_metrics, HH_data, checkpoint_dir)

            if eval_metrics.l2_total < best_l2
                best_l2 = eval_metrics.l2_total
                best_model_run = (model = trained_m, metrics = eval_metrics)
            end
        end

        mean_l2 = mean(seed_l2_errors)
        std_l2 = length(seed_l2_errors) > 1 ? std(seed_l2_errors) : 0.0
        mean_time = mean(seed_runtimes)

        println("    Params: $param_count | Rel L2 Error: $(round(mean_l2, digits=6)) | Time: $(round(mean_time, digits=2))s")

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

    output_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports"
    json_path = joinpath(output_dir, "benchmark_results.json")
    open(json_path, "w") do io
        JSON.print(io, results_dict, 4)
    end
    println("\n[SUCCESS] Single-seed (seed=42) benchmark complete! All weights saved to $checkpoint_dir")
    return results_dict
end

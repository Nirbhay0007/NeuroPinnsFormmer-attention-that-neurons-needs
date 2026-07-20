# ==============================================================================
# SYSTEM B: Activation & Spectral Benchmark Execution Script with Multi-Checkpointing
# Focus: M2 (tanh), M3 (GELU), M4 (SIREN-PINNsFormer), M-MLP, M-SIREN, M-Wang
# ==============================================================================
include("../benchmark_runner.jl")

println("=========================================================================")
println("LAUNCHING SYSTEM B: ACTIVATION & SPECTRAL BENCHMARKS")
println("Deterministic Seeds: $BENCHMARK_SEEDS")
println("=========================================================================\n")

data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
HH_data = CSV.read(data_path, DataFrame)
system_b_models = ["M2", "M3", "M4", "M-MLP", "M-SIREN", "M-Wang"]
results_system_b = Dict{String, Any}()
checkpoint_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports\checkpoints"

for id in system_b_models
    println("--> [System B] Benchmarking $id ...")
    dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
    
    seed_l2 = Float32[]
    seed_times = Float64[]
    
    for seed in BENCHMARK_SEEDS
        Random.seed!(seed)
        m_inst = instantiate_model(id, 10)
        trained_m, wall_t, history, best_w, milestones = train_model_single_run(m_inst, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
        metrics = evaluate_relative_l2_error(trained_m, HH_data)
        push!(seed_l2, metrics.l2_total)
        push!(seed_times, wall_t)
        
        # Save complete milestone & best/final checkpoint package
        save_model_checkpoints(id, seed, trained_m, best_w, milestones, history, metrics, HH_data, checkpoint_dir)
    end
    
    results_system_b[id] = Dict(
        "mean_l2" => mean(seed_l2),
        "std_l2" => std(seed_l2),
        "mean_runtime" => mean(seed_times)
    )
    println("    $id Mean L2 Error: $(round(mean(seed_l2), digits=6)) ± $(round(std(seed_l2), digits=6)) | Time: $(round(mean(seed_times), digits=2))s")
end

output_path = joinpath(dirname(@__DIR__), "..", "detailed_pinnsformmer_reports", "system_B_results.json")
open(output_path, "w") do io
    JSON.print(io, results_system_b, 4)
end
println("\n[SYSTEM B COMPLETE] Checkpoints saved to $checkpoint_dir")

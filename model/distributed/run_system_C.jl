# ==============================================================================
# SYSTEM C: Sequential Horizon Sweep Execution Script
# Focus: M6 (Context Window Sweeps k = 4, 8, 12, 16)
# ==============================================================================
include("../benchmark_runner.jl")

println("=========================================================================")
println("LAUNCHING SYSTEM C: SEQUENTIAL CONTEXT HORIZON SWEEPS")
println("Sweeping look-ahead horizon k in {4, 8, 12, 16} across 3 random seeds...")
println("=========================================================================\n")

data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
HH_data = CSV.read(data_path, DataFrame)
k_values = [4, 8, 12, 16]
results_system_c = Dict{String, Any}()

for k_val in k_values
    println("--> [System C] Benchmarking Horizon k = $k_val ...")
    dataloader, _, _ = prepare_dataloader(HH_data, k_val; batch_size=64)
    
    seed_l2 = Float32[]
    seed_times = Float64[]
    
    for seed in 1:3
        Random.seed!(42 + seed * 100)
        m_inst = PINNsFormer(; d_model=32, k=k_val, n_heads=4, depth=1, act_type=:wavelet)
        trained_m, wall_t = train_model_single_run(m_inst, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
        metrics = evaluate_relative_l2_error(trained_m, HH_data)
        push!(seed_l2, metrics.l2_total)
        push!(seed_times, wall_t)
    end
    
    key_str = "M6_k$(k_val)"
    results_system_c[key_str] = Dict(
        "horizon_k" => k_val,
        "mean_l2" => mean(seed_l2),
        "std_l2" => std(seed_l2),
        "mean_runtime" => mean(seed_times)
    )
    println("    k=$k_val Mean L2 Error: $(round(mean(seed_l2), digits=6)) ± $(round(std(seed_l2), digits=6)) | Time: $(round(mean(seed_times), digits=2))s")
end

output_path = joinpath(dirname(@__DIR__), "..", "detailed_pinnsformmer_reports", "system_C_results.json")
open(output_path, "w") do io
    JSON.print(io, results_system_c, 4)
end
println("\n[SYSTEM C COMPLETE] Results saved to $output_path")

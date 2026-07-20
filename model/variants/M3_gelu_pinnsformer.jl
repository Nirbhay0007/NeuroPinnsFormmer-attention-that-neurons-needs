# ==============================================================================
# Model M3: GELU-PINNsFormer Architecture (Transformer Baseline Activation)
# Replaces Wavelets with Standard GELU Activation Functions
# ==============================================================================
include("../architectures.jl")
include("../loss_and_optimization.jl")
include("../benchmark_runner.jl")

println("=== Model M3: GELU-PINNsFormer ===")
data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
HH_data = CSV.read(data_path, DataFrame)
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)

model_m3 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:gelu)
println("Parameter Count: ", count_parameters(model_m3))

trained_m3, wall_time = train_model_single_run(model_m3, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics = evaluate_relative_l2_error(trained_m3, HH_data)

println("--> Training Finished in $(round(wall_time, digits=2)) seconds")
println("--> Relative L2 Error: $(round(metrics.l2_total, digits=6))")
println("--> Component MSE (V, m, h, n): ($(round(metrics.mse_V, digits=4)), $(round(metrics.mse_m, digits=6)), $(round(metrics.mse_h, digits=6)), $(round(metrics.mse_n, digits=6)))")

# ==============================================================================
# Model M1: Proposed Base Wavelet-PINNsFormer Architecture
# Localized Wavelet Activation + EMA Dynamic NTK Loss Balancing + Temporal Causality
# ==============================================================================
include("../architectures.jl")
include("../loss_and_optimization.jl")
include("../benchmark_runner.jl")

println("=== Model M1: Base Wavelet-PINNsFormer ===")
data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
HH_data = CSV.read(data_path, DataFrame)
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)

model_m1 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:wavelet)
println("Parameter Count: ", count_parameters(model_m1))

trained_m1, wall_time = train_model_single_run(model_m1, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics = evaluate_relative_l2_error(trained_m1, HH_data)

println("--> Training Finished in $(round(wall_time, digits=2)) seconds")
println("--> Relative L2 Error: $(round(metrics.l2_total, digits=6))")
println("--> Component MSE (V, m, h, n): ($(round(metrics.mse_V, digits=4)), $(round(metrics.mse_m, digits=6)), $(round(metrics.mse_h, digits=6)), $(round(metrics.mse_n, digits=6)))")

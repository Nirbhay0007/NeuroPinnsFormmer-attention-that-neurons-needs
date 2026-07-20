import json
import os

def create_julia_notebook(title, description, code_cells, save_path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                f"{description}\n"
            ]
        }
    ]
    
    for code_block in code_cells:
        lines = [line + "\n" for line in code_block.strip().split("\n")]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })
        
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Julia 1.10",
                "language": "julia",
                "name": "julia-1.10"
            },
            "language_info": {
                "file_extension": ".jl",
                "mimetype": "application/julia",
                "name": "julia",
                "version": "1.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created Jupyter Notebook: {save_path}")

common_imports = """
using Plots, Flux, NNlib, LinearAlgebra, Statistics, Random, CSV, DataFrames, Optim, Printf

include("architectures.jl")
include("loss_and_optimization.jl")
"""

data_loader_code = """
# Load Hodgkin-Huxley Synthetic Ground-Truth Data
file_path = raw"C:\\nirbhay\\Downloads\\NeuroPinnsFormmer-attention-that-neurons-needs\\Synthetic_Data\\HH_ground_truth_synthetic_data.csv"
HH_data = CSV.read(file_path, DataFrame)
first(HH_data, 5)
"""

# 1. Base Model M1 Notebook
create_julia_notebook(
    title="Model M1: Base Wavelet-PINNsFormer Architecture",
    description="Proposed Physics-Informed Transformer with Localized Wavelet Activation, EMA Dynamic NTK Weight Balancing, and Temporal Causality Mask for 4D Hodgkin-Huxley Dynamics.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
# Instantiate Model M1 (Base Wavelet-PINNsFormer)
model_m1 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:wavelet)
flat_params, _ = Flux.destructure(model_m1)
println("Total Model Parameters: ", length(flat_params))
""",
        """
# Prepare Dataloader & Run Dual-Stage Optimization (30k Adam + 500 L-BFGS)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)

trained_m1, wall_time = train_model_single_run(model_m1, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_m1 = evaluate_relative_l2_error(trained_m1, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) seconds")
println("--> Final Relative L2 State Error: $(round(metrics_m1.l2_total, digits=6))")
""",
        """
# Plot Membrane Voltage & Ion Channel Gating Dynamics
include("paper_plots.jl")
configure_paper_style()

t = HH_data.t
p_v = plot(t, HH_data.V, color=:black, line=:dash, label="Ground Truth (Radau)")
plot!(t, metrics_m1.V_pred, color=:crimson, label="Wavelet-PINNsFormer Prediction")
title!("Membrane Potential Dynamics V(t)")
ylabel!("Voltage (mV)")

p_g = plot(t, HH_data.m, color=:black, line=:dash, label="GT m")
plot!(t, metrics_m1.m_pred, color=:dodgerblue, label="Model m")
plot!(t, metrics_m1.h_pred, color=:darkorange, label="Model h")
plot!(t, metrics_m1.n_pred, color=:forestgreen, label="Model n")
title!("Ion Channel Gating Kinetics (m, h, n)")
xlabel!("Time (ms)")
ylabel!("Gating Probability")

dashboard = plot(p_v, p_g, layout=grid(2, 1, heights=[0.5, 0.5]), link=:x, size=(800, 600))
display(dashboard)
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M1_base_wavelet_pinnsformer.ipynb"
)

# 2. Model M2 (tanh PINNsFormer) Notebook
create_julia_notebook(
    title="Model M2: tanh-PINNsFormer Architecture (Activation Ablation)",
    description="PINNsFormer variant replacing localized Mexican Hat Wavelets with continuous global tanh activation functions to evaluate spectral bias mitigation.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
# Instantiate Model M2 (tanh Activation Variant)
model_m2 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:tanh)
println("Parameter Count: ", count_parameters(model_m2))
""",
        """
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_m2, wall_time = train_model_single_run(model_m2, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_m2 = evaluate_relative_l2_error(trained_m2, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_m2.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M2_tanh_pinnsformer.ipynb"
)

# 3. Model M3 (GELU PINNsFormer) Notebook
create_julia_notebook(
    title="Model M3: GELU-PINNsFormer Architecture (Transformer Baseline)",
    description="PINNsFormer variant using standard GELU activations to benchmark against modern deep learning transformer standards.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_m3 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:gelu)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_m3, wall_time = train_model_single_run(model_m3, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_m3 = evaluate_relative_l2_error(trained_m3, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_m3.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M3_gelu_pinnsformer.ipynb"
)

# 4. Model M4 (SIREN PINNsFormer) Notebook
create_julia_notebook(
    title="Model M4: SIREN-PINNsFormer Architecture (Sinusoidal Frequency Variant)",
    description="PINNsFormer variant with periodic sin(30*x) activation layers to evaluate whether raw frequency features resolve stiff upstrokes without spatial wavelets.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_m4 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:siren)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_m4, wall_time = train_model_single_run(model_m4, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_m4 = evaluate_relative_l2_error(trained_m4, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_m4.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M4_siren_pinnsformer.ipynb"
)

# 5. Model M5 (Static NTK Ablation) Notebook
create_julia_notebook(
    title="Model M5: Static Loss Weight PINNsFormer Architecture (-NTK Ablation)",
    description="PINNsFormer variant disabling the dynamic EMA-NTK weight balancing loop (fixing all lambda_i = 1.0) to isolate optimization impact.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_m5 = PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:wavelet)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_m5, wall_time = train_model_single_run(model_m5, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=false)
metrics_m5 = evaluate_relative_l2_error(trained_m5, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_m5.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M5_static_ntk_pinnsformer.ipynb"
)

# 6. External Baseline: Vanilla MLP PINN Notebook
create_julia_notebook(
    title="Model M-MLP: Vanilla MLP-PINN Baseline",
    description="Standard 4-layer Multilayer Perceptron PINN establishing traditional continuous activation baseline performance.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_mlp = VanillaMLP(; hidden_dim=64, num_layers=4, act_fn=tanh)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_mlp, wall_time = train_model_single_run(model_mlp, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_mlp = evaluate_relative_l2_error(trained_mlp, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_mlp.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M_mlp_pinn_baseline.ipynb"
)

# 7. External Baseline: SIREN Network Notebook
create_julia_notebook(
    title="Model M-SIREN: SIREN PINN Baseline",
    description="Implicit Neural Representation network mapping time via periodic sinusoidal activation functions sin(30.0 * (W*x + b)).",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_siren = SIREN_PINN(; hidden_dim=64, num_layers=4, omega0=30.0f0)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_siren, wall_time = train_model_single_run(model_siren, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_siren = evaluate_relative_l2_error(trained_siren, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_siren.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M_siren_pinn_baseline.ipynb"
)

# 8. External Baseline: Modified Wang Gated PINN Notebook
create_julia_notebook(
    title="Model M-Wang: Modified Gated PINN Baseline (Wang et al., 2021)",
    description="Specialized PINN incorporating twin forward scaling encoders and multiplicative gating layers to resolve gradient pathologies in standard MLPs.",
    code_cells=[
        common_imports,
        data_loader_code,
        """
model_wang = ModifiedMLP_PINN(; hidden_dim=64, num_layers=4)
include("benchmark_runner.jl")
dataloader, _, _ = prepare_dataloader(HH_data, 10; batch_size=64)
trained_wang, wall_time = train_model_single_run(model_wang, dataloader; adam_epochs=1000, lbfgs_epochs=200, use_ntk=true)
metrics_wang = evaluate_relative_l2_error(trained_wang, HH_data)

println("--> Training Completed in $(round(wall_time, digits=2)) s")
println("--> Relative L2 State Error: $(round(metrics_wang.l2_total, digits=6))")
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\M_wang_gated_pinn_baseline.ipynb"
)

# 9. Master Comparative Benchmark Notebook
create_julia_notebook(
    title="Master Comparative Benchmark & Paper Plotting Dashboard",
    description="Automated execution pipeline running full model taxonomy across multiple seeds, logging CPU execution times, peak RAM, parameter counts, and generating 5 publication-ready figures.",
    code_cells=[
        common_imports,
        """
include("benchmark_runner.jl")
include("paper_plots.jl")

# Execute benchmark suite across all model variants
results = run_benchmark_suite(
    adam_epochs = 500,
    lbfgs_epochs = 100,
    num_seeds = 3
)

# Generate publication plots in detailed_pinnsformmer_reports/plots/
output_dir = raw"C:\\nirbhay\\Downloads\\NeuroPinnsFormmer-attention-that-neurons-needs\\detailed_pinnsformmer_reports"
generate_all_paper_plots(results, output_dir)
"""
    ],
    save_path=r"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\model\master_comparative_benchmark.ipynb"
)

print("[COMPLETE] Created all 9 Jupyter Notebooks successfully!")

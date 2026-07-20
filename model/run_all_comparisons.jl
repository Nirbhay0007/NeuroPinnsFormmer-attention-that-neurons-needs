# ==============================================================================
# Master Benchmark Runner & Paper Plot Generator
# ==============================================================================
include("benchmark_runner.jl")
include("paper_plots.jl")

println("=========================================================================")
println("EXECUTING FULL MODEL COMPARISON PIPELINE FOR SCIENTIFIC PAPER")
println("=========================================================================\n")

# Run full comparative benchmark suite across model taxonomy
results = run_benchmark_suite(
    adam_epochs = 1000,
    lbfgs_epochs = 200,
    num_seeds = 3
)

# Generate publication-grade scientific figures
output_dir = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\detailed_pinnsformmer_reports"
generate_all_paper_plots(results, output_dir)

println("\n[COMPLETED] Full model comparison suite finished successfully!")

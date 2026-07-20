# ==============================================================================
# Model Weight Loader & Instant Prediction / Audit Script (NO RE-TRAINING)
# ==============================================================================
include("benchmark_runner.jl")
include("paper_plots.jl")

"""
    audit_checkpoint(model_id::String, seed::Int=101, checkpoint_type::String="best")

Loads pre-trained model weights from disk and evaluates forward pass predictions
instantly without re-training.
"""
function audit_checkpoint(model_id::String, seed::Int=101, checkpoint_type::String="best")
    println("=========================================================================")
    println("LOADING CHECKPOINT [$model_id | Seed: $seed | Type: $checkpoint_type]")
    println("=========================================================================")

    data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
    HH_data = CSV.read(data_path, DataFrame)
    k_val = model_id == "M6" ? 8 : 10

    # Inject saved weights into fresh architecture shell
    restored_model, meta = load_model_checkpoint(model_id, seed; checkpoint_type=checkpoint_type, k_val=k_val)

    # Compute immediate predictions
    metrics = evaluate_relative_l2_error(restored_model, HH_data)

    println("--> Total Relative L2 Error : ", round(metrics.l2_total, digits=6))
    println("--> Voltage MSE (V)         : ", round(metrics.mse_V, digits=6))
    println("--> Gating MSE (m, h, n)    : ($(round(metrics.mse_m, digits=6)), $(round(metrics.mse_h, digits=6)), $(round(metrics.mse_n, digits=6)))")

    # Plot predictions vs ground truth
    configure_paper_style()
    t = HH_data.t
    p_v = plot(t, HH_data.V, color=:black, line=:dash, label="Ground Truth (Radau)")
    plot!(t, metrics.V_pred, color=:crimson, label="$model_id Restored Prediction")
    title!("Restored Checkpoint Membrane Potential V(t)")
    ylabel!("Voltage (mV)")

    p_g = plot(t, HH_data.m, color=:black, line=:dash, label="GT m")
    plot!(t, metrics.m_pred, color=:dodgerblue, label="Restored m")
    plot!(t, metrics.h_pred, color=:darkorange, label="Restored h")
    plot!(t, metrics.n_pred, color=:forestgreen, label="Restored n")
    title!("Restored Checkpoint Ion Channel Kinetics")
    xlabel!("Time (ms)")
    ylabel!("Gating Probability")

    audit_plot = plot(p_v, p_g, layout=grid(2, 1, heights=[0.5, 0.5]), link=:x, size=(800, 600))
    display(audit_plot)

    return restored_model, metrics
end

# Example usage (Uncomment to test loading M1 base model):
# restored_m1, metrics = audit_checkpoint("M1", 101, "best")

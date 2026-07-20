# ==============================================================================
# Publication Visualization Suite for Hodgkin-Huxley PINNsFormer Paper
# ==============================================================================
using Plots
using CSV
using DataFrames
using JSON

"""
    configure_paper_style()

Sets up Computer Modern typography, high DPI, dash gridlines, and clean legend boxes.
"""
function configure_paper_style()
    default(
        fontfamily = "Computer Modern",
        titlefontsize = 12,
        guidefontsize = 11,
        tickfontsize = 9,
        legendfontsize = 9,
        grid = true,
        gridalpha = 0.15,
        gridstyle = :dash,
        frame = :box,
        lw = 2.0
    )
end

"""
    generate_all_paper_plots(benchmark_results_dict::Dict, output_dir::String)

Reads dataset and benchmark outputs, creating and saving 5 publication figures.
"""
function generate_all_paper_plots(results_dict::Dict, output_dir::String)
    configure_paper_style()
    mkpath(output_dir)
    plots_dir = joinpath(output_dir, "plots")
    mkpath(plots_dir)

    data_path = raw"C:\nirbhay\Downloads\NeuroPinnsFormmer-attention-that-neurons-needs\Synthetic_Data\HH_ground_truth_synthetic_data.csv"
    HH_data = CSV.read(data_path, DataFrame)
    t = HH_data.t

    # Extract model predictions safely
    m1_pred = get(results_dict, "M1", Dict())["best_predictions"]
    m2_pred = get(results_dict, "M2", Dict())["best_predictions"]
    m_mlp_pred = get(results_dict, "M-MLP", Dict())["best_predictions"]

    # --------------------------------------------------------------------------
    # Plot 1: Predictive Trajectory Overlay (Fidelity Chart)
    # --------------------------------------------------------------------------
    println("--> Generating Plot 1: Predictive Trajectory Overlay...")
    p1_v = plot(t, HH_data.V, color=:black, line=:dash, label="Ground Truth (Radau)")
    if !isnothing(m1_pred)
        plot!(t, m1_pred["V"], color=:crimson, label="Ours (Wavelet-PINNsFormer)")
    end
    if !isnothing(m2_pred)
        plot!(t, m2_pred["V"], color=:dodgerblue, linestyle=:dot, label="PINNsFormer (tanh)")
    end
    if !isnothing(m_mlp_pred)
        plot!(t, m_mlp_pred["V"], color=:darkorange, linestyle=:dashdot, label="Vanilla MLP-PINN")
    end
    title!("Membrane Potential Dynamics V(t)")
    ylabel!("Voltage (mV)")

    p1_gates = plot(t, HH_data.m, color=:black, line=:dash, label="GT m")
    if !isnothing(m1_pred)
        plot!(t, m1_pred["m"], color=:dodgerblue, label="Ours m")
        plot!(t, m1_pred["h"], color=:darkorange, label="Ours h")
        plot!(t, m1_pred["n"], color=:forestgreen, label="Ours n")
    end
    title!("Ion Channel Gating Kinetics (m, h, n)")
    xlabel!("Time (ms)")
    ylabel!("Gating Probability")

    plot1 = plot(p1_v, p1_gates, layout=grid(2, 1, heights=[0.5, 0.5]), link=:x, size=(850, 600), dpi=300)
    savefig(plot1, joinpath(plots_dir, "plot1_trajectory_overlay.png"))

    # --------------------------------------------------------------------------
    # Plot 2: Spatiotemporal Absolute Error Evolution
    # --------------------------------------------------------------------------
    println("--> Generating Plot 2: Spatiotemporal Absolute Error Evolution...")
    if !isnothing(m1_pred) && !isnothing(m_mlp_pred)
        err_m1_V = abs.(m1_pred["V"] .- HH_data.V)
        err_mlp_V = abs.(m_mlp_pred["V"] .- HH_data.V)

        p2 = plot(t, err_m1_V, color=:crimson, label="Ours (Wavelet-PINNsFormer)")
        plot!(t, err_mlp_V, color=:darkorange, linestyle=:dash, label="Vanilla MLP-PINN")
        title!("Absolute Error Evolution |V_pred - V_true|")
        xlabel!("Time (ms)")
        ylabel!("Absolute Error (mV)")
        savefig(p2, joinpath(plots_dir, "plot2_error_evolution.png"))
    end

    # --------------------------------------------------------------------------
    # Plot 4: Quantitative Performance Bar Chart Across Model Taxonomy
    # --------------------------------------------------------------------------
    println("--> Generating Plot 4: Comparative Relative L2 Error Bar Chart...")
    model_ids = ["M1", "M2", "M3", "M4", "M5", "M7", "M-MLP", "M-SIREN", "M-Wang"]
    labels = String[]
    l2_means = Float64[]
    l2_stds = Float64[]

    for id in model_ids
        if haskey(results_dict, id)
            push!(labels, id)
            push!(l2_means, results_dict[id]["mean_l2_error"])
            push!(l2_stds, results_dict[id]["std_l2_error"])
        end
    end

    if !isempty(l2_means)
        p4 = bar(
            labels, l2_means, 
            yerror = l2_stds, 
            yscale = :log10, 
            color = :dodgerblue,
            legend = false,
            title = "Model Taxonomy Relative L2 State Tracking Error",
            xlabel = "Model Architecture ID",
            ylabel = "Relative L2 Error (Log Scale)",
            size = (800, 450),
            dpi = 300
        )
        savefig(p4, joinpath(plots_dir, "plot4_ablation_comparison_barchart.png"))
    end

    println("[SUCCESS] All paper visualization charts saved to $plots_dir")
end

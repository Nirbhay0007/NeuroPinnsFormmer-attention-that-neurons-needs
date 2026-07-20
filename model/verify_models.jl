# ==============================================================================
# Model Architecture & Loss Vectorization Verification Script
# ==============================================================================
include("architectures.jl")
include("loss_and_optimization.jl")

println("=========================================================================")
println("RUNNING VERIFICATION SUITE FOR ALL MODELS & LOSS ENGINE")
println("=========================================================================\n")

# Mock inputs
batch_size = 16
k = 10
t_batch_3d = rand(Float32, 1, 1, batch_size)
t_batch_seq = rand(Float32, 1, k, batch_size)
gt_batch = rand(Float32, 4, k, batch_size)
I_ext = 10.0f0

models_to_test = [
    ("M1 (PINNsFormer Wavelet)", PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:wavelet)),
    ("M2 (PINNsFormer tanh)", PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:tanh)),
    ("M3 (PINNsFormer GELU)", PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:gelu)),
    ("M4 (PINNsFormer SIREN)", PINNsFormer(; d_model=32, k=10, n_heads=4, depth=1, act_type=:siren)),
    ("M7 (PINNsFormer Capacity)", PINNsFormer(; d_model=16, k=10, n_heads=2, depth=1, act_type=:wavelet)),
    ("M-MLP (Vanilla MLP)", VanillaMLP(; hidden_dim=64, num_layers=4, act_fn=tanh)),
    ("M-SIREN (SIREN Network)", SIREN_PINN(; hidden_dim=64, num_layers=4, omega0=30.0f0)),
    ("M-Wang (Modified Wang MLP)", ModifiedMLP_PINN(; hidden_dim=64, num_layers=4))
]

for (name, model) in models_to_test
    print("Testing $name ... ")
    # Test forward pass
    out = model(t_batch_3d)
    @assert size(out, 1) == 4 "Output channel dimension must be 4"
    
    # Test derivative extraction
    X_pred, dX_dt = compute_ad_derivatives(model, t_batch_3d)
    @assert size(X_pred) == size(dX_dt) "Prediction and derivative shapes must match"
    
    # Test loss calculation
    λ_state = NTKState()
    loss = total_loss_causal(model, t_batch_3d, gt_batch, λ_state, I_ext)
    @assert !isnan(loss) && !isinf(loss) "Loss must be finite"
    
    println("PASS! Output shape: $(size(out)) | Loss: $(round(loss, digits=4))")
end

println("\n[ALL MODELS VERIFIED SUCCESSFULLY!]")

# Project Plan: Implementation of PINNsFormer for the 4D Hodgkin-Huxley (HH) Neuron Model  
This tactical research roadmap outlines the exact phases, architectural variations, optimization pipelines, and experimental benchmarks required to elevate this project into a publication-quality Scientific Machine Learning (SciML) research work .

---

## Phase 1: Ground-Truth Data Generation & Environment Setup  
Before assembling the neural network, you must establish an absolute reference framework using a trusted numerical baseline to validate the model and diagnose errors .

* **Select a Stiff ODE Solver:** Implement the classical 4D Hodgkin-Huxley system using a high-order stiff solver (such as Radau) . This is critical to safely handle the extreme spike-induced stiffness ($S_{\text{ratio}} > 10^4 - 10^6$) that occurs during action potential upstrokes .
* **Establish the State Vector Baseline:** Define the continuous physical state vector explicitly as $$\mathbf{x}(t) = [V(t), m(t), h(t), n(t)]^T \in \mathbb{R}^4$$ where $V$ is membrane voltage, $m$ is sodium activation, $h$ is sodium inactivation, and $n$ is potassium activation .
* **Generate Three Primary Stimulus Profiles:**
    * **Profile A (Single Spike):** Apply a short current pulse ($I_{\text{ext}} = 10.0\,\mu\text{A/cm}^2$) for $t \in [0, 2.0\,\text{ms}]$ over a total window of $20.0\,\text{ms}$ .
    * **Profile B (Tonic Spiking):** Apply a sustained DC current ($I_{\text{ext}} = 15.0\,\mu\text{A/cm}^2$) past the subcritical Andronov-Hopf bifurcation point to induce stable limit cycle oscillations for $100.0\,\text{ms}$ .
    * **Profile C (Bursting Dynamics):** Create a multi-scale slow-harmonic current profile ($I_{\text{ext}}(t) = 5.0 \sin(0.1t) + 8.0\,\mu\text{A/cm}^2$) to drive alternating periods of spiking and quiescence over a $500\,\text{ms}$ window .

---

## Phase 2: Building Core PINNsFormer Structural Components  
Instead of pointwise coordinate evaluation, your architecture will map temporal coordinates into continuous short trajectories .

### 1. Pseudo-Sequence Generation Layer  
* For each input temporal coordinate $t$, construct a marching forward-looking vector of length $k$ using a fixed look-ahead step size $\Delta t$, such that $t_\gamma = t + \gamma \Delta t$ for $\gamma = 0, \dots, k-1$ .
* Superimpose sinusoidal positional encodings onto the linear input mixer projection to preserve temporal ordering across the sequence .

### 2. Learnable Wavelet Activations  
* Replace standard activation functions (like ReLU or Tanh) within the Transformer feedforward layers with a custom periodic activation governed by the equation :
$$\text{Wavelet}(z) = \omega_1 \sin(z) + \omega_2 \cos(z)$$
* This introduces a localized Fourier mode effect, which resolves high-frequency localized upstroke spikes and mitigates the classical network spectral bias .

### 3. Core Architecture: Fast-Slow Decomposition (Variant D)  
To maximize research novelty, implement Variant D (Hierarchical Timescale PINNsFormer), which mirrors the biological segregation of ultra-fast activator dynamics ($V, m$) and slow recovery pathways ($h, n$) :

| Path Setup | State Variables Assigned | Temporal Resolution | Architectural Role |
| :--- | :--- | :--- | :--- |
| **Fine-Timescale Path** | $V, m$   | $\Delta t = 0.05\,\text{ms}$   | Resolves rapid action potential upstrokes . |
| **Coarse-Timescale Path** | $h, n$   | $\Delta t = 0.5\,\text{ms}$   | Captures slow post-spike recovery decay . |

> **Alignment Note:** Align these two parallel processing paths using a linear interpolation operator before routing them to the final multi-variable prediction projection head .

---

## Phase 3: Physics Loss & Two-Stage Optimization Pipeline  

### 1. Formulate the Objective Function  
Establish a composite sequential loss covering the look-ahead steps :
$$\mathcal{L}_{\text{total}} = \lambda_V \mathcal{L}_{\mathcal{R}V} + \lambda_m \mathcal{L}_{\mathcal{R}m} + \lambda_h \mathcal{L}_{\mathcal{R}h} + \lambda_n \mathcal{L}_{\mathcal{R}n} + \lambda_{\text{ic}}\mathcal{L}_{\text{ic}} + \lambda_{\text{data}}\mathcal{L}_{\text{data}}$$

* **Evaluate Sequential Residuals:** Compute exact temporal derivatives across your moving window using automatic differentiation (AD) on the network's outputs, enforcing the physical vector field equations consistently across the chain .
* **Neural Tangent Kernel (NTK) Weighting:** Neutralize gradient pathologies where the ODE residuals overpower the boundary constraints . Dynamically recompute the loss weights ($\lambda_i$) using the trace of the NTK at fixed epoch intervals .

### 2. Execute Two-Stage Optimization  
* **Stage 1 (Global Exploration):** Train with the Adam optimizer for a few thousand epochs to bypass local traps and approximate the broader physical manifold .
* **Stage 2 (Local Precision):** Transition to the second-order quasi-Newton L-BFGS optimizer to quickly lock onto exact high-frequency spike geometries with high numerical precision .

---

## Phase 4: Systematic Experimental Suite & Benchmarks  
To build a bulletproof empirical narrative against vanilla PINNs, run this standardized evaluation suite :

* **Experiment 1: Forward Single-Spike Accuracy**
    * *Setup:* Test on a $20.0\,\text{ms}$ window using Profile A to confirm the model does not damp or over-smooth the action potential peak .
    * *Metrics:* Compute relative Root Mean Square Error ($\text{rRMSE}_V$) and absolute peak error ($\Delta V_{\text{peak}}$) .
* **Experiment 2: Limit Cycle & Phase Drift Control**
    * *Setup:* Run the tonic spiking configuration (Profile B) over a long $100.0\,\text{ms}$ timeline .
    * *Metrics:* Track the predicted firing frequency error and check for phase drift over time using the Mean Absolute Error of Inter-Spike Intervals ($\text{MAE}_{\text{ISI}}$) .
* **Experiment 3: Inverse Biophysical Parameter Estimation**
    * *Setup:* Hide the maximal conductances ($\bar{g}_{\text{Na}}, \bar{g}_{\text{K}}, g_{\text{L}}$) and treat them as learnable parameters inside the network loss while providing sparse voltage observations .
    * *Metrics:* Reconstruct unobservable trajectories and calculate the percentage parameter recovery error : $$\text{Error}_{\theta} = \frac{|\theta_{\text{true}} - \theta_{\text{pred}}|}{\theta_{\text{true}}} \times 100\%$$
* **Experiment 4: Measurement Noise Invalidation**
    * *Setup:* Corrupt observed voltage measurements with additive Gaussian noise across discrete signal-to-noise ratios ($\text{SNR} \in \{10\,\text{dB}, 20\,\text{dB}, 30\,\text{dB}\}$) .
    * *Metrics:* Map sensitivity curves to show how gracefully your AD residuals degrade under noise .
* **Experiment 5: Long-Horizon Extrapolation**
    * *Setup:* Train exclusively on a short window ($t \in [0, 20.0\,\text{ms}]$) and evaluate performance as you extrapolate out to $100.0\,\text{ms}$ without internal residual tracking .

---

## Phase 5: Post-Hoc Analysis & Paper Positioning  

### The Research "Hook"  
Do not simply state that PINNsFormer works better. Prove *why* it works better by exposing the learned physical properties of the multi-head self-attention matrix ($\mathbf{A}_{\gamma, \gamma'}$) . You will extract and plot attention maps across distinct biophysical phases :

1. **Upstroke Phase:** Show that attention strictly limits its context window and concentrates directly along the main diagonal to cleanly isolate fast localized transitions .
2. **Repolarization Phase:** Document the asymmetric backward temporal drift, showing how the model queries historical states to compute temporal causality .
3. **Quiescence Phase:** Highlight the uniform, flat attention profile used to safely gather low-frequency equilibrium properties .

> **Core Thesis:** Frame your final analysis around this mechanism: self-attention operates as a self-adaptive variable time-step integrator that automatically changes its context window based on localized system stiffness . This allows it to smooth the optimization landscape where pointwise MLPs collapse .

### Rigorous Academic Defense of Synthetic Data  
When presenting to advisors or reviewers, frame the choice of synthetic data (from the Radau solver) not as a limitation, but as a strict, mathematically rigorous requirement for SciML validation :

* **Partial Observability:** Real experiments only record voltage ($V$) . The microscopic gating variables ($m, h, n$) are physically impossible to measure directly . Synthetic data provides a complete state vector to prove the network isn't just hallucinating hidden profiles that superficially fit $V(t)$ .
* **Structural Mismatch Isolation:** Biological cells introduce complex real-world anomalies like metabolic pump limits or temperature drifts . Synthetic data guarantees a perfect mathematical match, isolating architecture performance from modeling inadequacies .
* **Safeguarding AD Engines:** Real data is packed with high-frequency instrumentation noise . Taking exact automatic differentiation derivatives of uncurated noise causes physics residuals to explode . Synthetic data acts as a controlled sandbox to safely locate the exact breaking point of the model's AD engine .
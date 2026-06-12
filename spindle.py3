#!/usr/bin/env python3
"""
Deep Learning Analysis of 15N Relaxation Data

This script takes experimental R1, R2, and NOE data and uses an ensemble 
of deep neural networks to predict model-free parameters (S2, tauE, Rex).

Features:
- Ensemble Uncertainty Quantification (Error Bars)
- Automated "Bad Data" Detection (Ambiguity Index)
- Conformational Entropy Estimation
- Multi-Field Support

Usage:
    python predict_experimental_data.py3 data.txt --field 850 --out results_ubiquitin

Input Format (Space separated text file):
    # Residue  R1      R2      NOE
    1          1.2     12.5    0.78
    2          1.3     11.8    0.82
    ...
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from spindle_calibrations import CALIBRATIONS, MULTIPLIERS

# --- CONSTANTS & CONFIG ---
# Force CPU for inference (often faster for small batches & more stable for end-users)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Thresholds for Quality Control
AMBIGUITY_THRESHOLD_S2 = 0.031    # If ensemble std > 0.05, flag as ambiguous
AMBIGUITY_THRESHOLD_TAUE = 47.03  # If ensemble std > 2.0 s^-1, flag as ambiguous
AMBIGUITY_THRESHOLD_REX = 2.45    # If ensemble std > 2.0 s^-1, flag as ambiguous

# --- FUNCTIONS ---
def parse_input_file(filepath):
    """
    Parses a simple 4-column text file: Residue R1 R2 NOE
    Skips lines starting with # or @
    """
    residues = []
    data = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('#', '@')):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                # Check if first col is a number (residue ID)
                try:
                    res_id = int(parts[0])
                    r1 = float(parts[1])
                    r2 = float(parts[2])
                    noe = float(parts[3])
                    
                    residues.append(res_id)
                    data.append([r1, r2, noe])
                except ValueError:
                    continue # Skip header lines that aren't comments
                    
        return np.array(residues), np.array(data)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

def load_ensemble(field_mhz, base_model_dir="models"):
    """
    Loads the correct ensemble based on field strength.
    Expects structure: models/850/model_0.keras
    """
    # Construct path (e.g., ./models/850)
    # You can change the naming convention here
    model_dir = os.path.join(base_model_dir, str(field_mhz))
    
    if not os.path.exists(model_dir):
        print(f"Error: No models found for {field_mhz} MHz in {model_dir}")
        print("Available fields might be:", [d for d in os.listdir(base_model_dir) if os.path.isdir(os.path.join(base_model_dir, d))])
        sys.exit(1)
        
    models = []
    print(f"Loading ensemble from {model_dir}...")
    
    # Look for .keras files
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
    
    if not model_files:
        print("Error: No .keras files found in directory.")
        sys.exit(1)
        
    for m_file in model_files:
        full_path = os.path.join(model_dir, m_file)
        try:
            # compile=False is safer for inference across TF versions
            model = tf.keras.models.load_model(full_path, compile=False) 
            models.append(model)
        except Exception as e:
            print(f"Warning: Failed to load {m_file}: {e}")
            
    print(f"Successfully loaded {len(models)} models.")
    return models

def calculate_entropy(s2):
    """
    Estimates conformational entropy dS_conf from S2 using the 
    Frederick approximation (or similar linear proxy).
    dS (cal/mol/K) approx -k_B * ...
    Using a simplified linear proxy for illustration: dS ~ (1 - S2) * Scale
    """
    # Simple proxy: Rigid (1.0) -> 0 entropy. Flexible (0.5) -> High entropy.
    # This is a placeholder for the exact Frederick/Yang formula
    return (1.0 - s2) * 5.0 # Arbitrary units for demo

def apply_calibration(pred_mean, pred_err, calib, mult):
    """Applies the polynomial corrections derived from synthetic testing"""
    # S2
    s2_corr = (pred_mean[:, 1] * calib['S2']['slope']) + calib['S2']['intercept']
    s2_corr = np.clip(s2_corr, 0.0, 1.0)
    s2_corr_err = pred_err[:,1] / calib['S2']['slope'] * mult['S2']['mult']
    
    # tauE
    tauE_ps = pred_mean[:, 0] * 1000.0
    tauE_corr_ps = (tauE_ps * calib['TAUE']['slope']) + calib['TAUE']['intercept']
    tauE_corr_ps = np.maximum(tauE_corr_ps, 0)
    tauE_corr_err = pred_err[:,0] * 1000.0 / calib['TAUE']['slope'] * mult['TAUE']['mult']
    
    # Rex
    rex = pred_mean[:, 2]
    rex_corr = (calib['REX']['a'] * rex**2) + (calib['REX']['b'] * rex) + calib['REX']['c']
    rex_corr = np.maximum(rex_corr, 0.0)

    rex_derivative = np.abs(2 * calib['REX']['a'] * rex + calib['REX']['b'])
    rex_derivative = np.where(rex_derivative == 0, 1e-6, rex_derivative) # Avoid divide by zero
    rex_corr_err = pred_err[:,2] / rex_derivative * mult['REX']['mult']
    
    return np.stack([tauE_corr_ps, s2_corr, rex_corr], axis=1), np.stack([tauE_corr_err, s2_corr_err, rex_corr_err], axis=1)

def generate_dashboard(residues, predictions, errors, s2_amb, taue_amb, rex_amb, global_tauc, output_prefix):
    """Creates a publication-ready figure"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Order Parameter S2
    # Color code points by Quality: Blue=Good, Red=Ambiguous
    colors_s2 = ['red' if amb else 'black' for amb in s2_amb]
    colors_taue = ['red' if amb else 'black' for amb in taue_amb]
    colors_rex = ['red' if amb else 'black' for amb in rex_amb]
    
    axes[0].errorbar(residues, predictions[:, 1], yerr=errors[:, 1], fmt='none', ecolor='gray', alpha=0.5)
    axes[0].scatter(residues, predictions[:, 1], c=colors_s2, s=20)
    axes[0].set_ylabel(r"$S^2$ (Order Parameter)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Global Tumbling $\\tau_c$: {global_tauc:.2f} ns", loc='right')
    
    # 2. Internal Timescale tauE
    axes[1].errorbar(residues, predictions[:, 0], yerr=errors[:, 0], fmt='none', ecolor='gray', alpha=0.5)
    axes[1].scatter(residues, predictions[:, 0], c=colors_taue, s=20)
    axes[1].set_ylabel(r"$\tau_E$ (ps)")
    # Log scale is often better for tauE if it spans orders of magnitude, but linear is fine for now
    
    # 3. Exchange Rex
    axes[2].errorbar(residues, predictions[:, 2], yerr=errors[:, 2], fmt='none', ecolor='gray', alpha=0.5)
    axes[2].scatter(residues, predictions[:, 2], c=colors_rex, s=20)
    axes[2].set_ylabel(r"$R_{ex}$ ($s^{-1}$)")
    axes[2].set_xlabel("Residue Index")
    
    # Add a legend for the red dots
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='High Confidence'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Ambiguous Data')]
    axes[0].legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_dashboard.png", dpi=300)
    print(f"Dashboard saved to {output_prefix}_dashboard.png")

def extract_attention_weights(models, input_data):
    """
    Runs the input through the ensemble to extract the attention scores.
    Returns: Average attention weights (normalized 0-1) across the ensemble.
    """
    print("Extracting Attention/Relevance scores from the neural network...")

    # 1. Find the attention layer by class or naming convention
    target_layer = None
    test_model = models[0]

    for layer in test_model.layers:
        if isinstance(layer, tf.keras.layers.Attention) or 'attention' in layer.name.lower():
            target_layer = layer
            break

    if not target_layer:
        print("Warning: Could not auto-detect Attention layer. Skipping explanation plots.")
        return None

    # 2. Extract weights from all models retroactively
    all_weights = []
    for model in models:
        # Get the specific model's attention layer instance
        layer_instance = model.get_layer(target_layer.name)

        # Build an intermediate model that outputs the *inputs* to the attention layer
        inter_model = tf.keras.Model(inputs=model.input, outputs=layer_instance.input)
        layer_inputs_val = inter_model.predict(input_data, verbose=0)

        # Convert inputs to eager tensors so we can execute the layer directly
        if isinstance(layer_inputs_val, list):
            layer_inputs_tensor = [tf.convert_to_tensor(v) for v in layer_inputs_val]
        else:
            layer_inputs_tensor = tf.convert_to_tensor(layer_inputs_val)

        # Call the attention layer instance eagerly, forcing it to return the scores matrix
        _, att_scores = layer_instance(layer_inputs_tensor, return_attention_scores=True)

        # att_scores has shape (Batch, Queries, Keys) -> (1, N_residues, N_residues)
        scores_np = att_scores.numpy().squeeze(axis=0) # Now shape (N_residues, N_residues)

        # The 2D matrix tells us how much attention residue i pays to residue j.
        # Averaging along axis 0 gives the global importance/relevance score for each residue.
        importance_per_residue = np.mean(scores_np, axis=0)
        all_weights.append(importance_per_residue)

    # 3. Average across the ensemble and Normalize
    avg_weights = np.mean(all_weights, axis=0)

    # Min-Max Normalization (0 to 1) for the explainability plots
    norm_weights = (avg_weights - np.min(avg_weights)) / (np.max(avg_weights) - np.min(avg_weights))

    return norm_weights

def plot_mechanism_explanation(residues, attention, s2, rex, output_prefix):
    """
    Plots Attention (AI Brain) vs S2 (Rigidity) and Rex (Exchange).
    Validates the hypothesis: High Attention = High S2 + Low Rex
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Panel 1: Attention vs S2 ---
    color_att = 'tab:gray'
    color_s2 = 'tab:blue'
    
    # Plot Attention as a filled area in the background
    ax1.fill_between(residues, 0, attention, color=color_att, alpha=0.3, label='AI Attention Score')
    ax1.set_ylabel('Attention (Importance)', color=color_att, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_att)
    
    # Create twin axis for S2
    ax1_twin = ax1.twinx()
    ax1_twin.plot(residues, s2, color=color_s2, linewidth=2, label='Predicted S2')
    ax1_twin.set_ylabel(r'Order Parameter ($S^2$)', color=color_s2, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_s2)
    ax1_twin.set_ylim(0, 1.1)
    
    ax1.set_title("Why did the model choose this tau_c?", loc='left', fontsize=12)
    ax1.text(0.02, 0.95, "Hypothesis: Model focuses on Rigid Residues", transform=ax1.transAxes, fontsize=9, style='italic')

    # --- Panel 2: Attention vs Rex ---
    color_rex = 'tab:red'
    
    # Plot Attention again for reference
    ax2.fill_between(residues, 0, attention, color=color_att, alpha=0.3, label='AI Attention Score')
    ax2.set_ylabel('Attention (Importance)', color=color_att, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_att)
    
    # Create twin axis for Rex
    ax2_twin = ax2.twinx()
    # Use stems/bars for Rex since it's sparse
    ax2_twin.bar(residues, rex, color=color_rex, alpha=0.6, width=0.8, label='Predicted Rex')
    ax2_twin.set_ylabel(r'$R_{ex} (s^{-1})$', color=color_rex, fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=color_rex)
    
    ax2.set_xlabel("Residue Index")
    ax2.text(0.02, 0.95, "Hypothesis: Model ignores Exchange Residues", transform=ax2.transAxes, fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_explainability.png", dpi=300)
    print(f"Explanation plot saved to {output_prefix}_explainability.png")

def main():
    parser = argparse.ArgumentParser(description="AI-Driven ModelFree Analysis")
    parser.add_argument("-i", "--input_file", help="Path to text file with columns: Residue_number R1 R2 NOE")
    parser.add_argument("-f", "--field", type=int, default=850, help="Proton frequency in MHz (default: 850)")
    parser.add_argument("-o", "--out", type=str, default="output", help="Prefix for output files")
    parser.add_argument("-m", "--model_dir", type=str, default="./models", help="Base directory containing model subfolders")
    parser.add_argument("-e", "--explain", action="store_true", help="Flag to generate Attention plots to explain model decisions")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()

    if args.field not in CALIBRATIONS:
        print(f"ERROR: No calibration data found for {args.field} MHz.")
        print(f"Available fields: {list(CALIBRATIONS.keys())}")
        sys.exit(1)

    calib = CALIBRATIONS[args.field]
    
    if args.field not in MULTIPLIERS:
        print(f"ERROR: No calibration data found for {args.field} MHz.")
        print(f"Available fields: {list(MULTIPLIERS.keys())}")
        sys.exit(1)

    error_mult = MULTIPLIERS[args.field]
    
    # 1. Load Data
    print(f"Reading {args.input_file}...")
    residues, data = parse_input_file(args.input_file)
    print(f"Found {len(residues)} residues.")
    
    # 2. Load AI Models
    models = load_ensemble(args.field, args.model_dir)
    
    # 3. Run Inference
    # Data shape needs to be (1, N_res, 3) for the model
    input_tensor = np.expand_dims(data, axis=0).astype(np.float32)
    
    print("Running ensemble inference...")
    tauc_preds = []
    local_preds = []
    
    for i, model in enumerate(models):
        # Predict
        l_pred, g_pred = model.predict(input_tensor, verbose=0)
        
        # l_pred shape is likely (1, N_res, 3)
        # g_pred shape is likely (1, 1)
        tauc_preds.append(g_pred[0, 0])
        local_preds.append(l_pred[0]) # Remove batch dim
        
    # 4. Process Ensemble Statistics
    tauc_mean = np.mean(tauc_preds)
    tauc_std  = np.std(tauc_preds)
    
    local_preds = np.array(local_preds) # Shape: (N_models, N_res, 3)
    
    # Calculate Mean and Uncertainty (Std Dev)
    local_mean = np.mean(local_preds, axis=0) # Shape: (N_res, 3)
    local_err  = np.std(local_preds, axis=0)

    # Correct tauc
    tauc_mean = (tauc_mean - calib['TAUC']['intercept']) / calib['TAUC']['slope']
    tauc_std = tauc_std / calib['TAUC']['slope'] * error_mult['TAUC']['mult']
    
    # 5. Apply Calibration (Bias Correction)
    local_corrected_mean, local_corrected_err = apply_calibration(local_mean, local_err, calib, error_mult)
    
    # 6. Quality Control & Derived Metrics (Decoupled Parameter Tracking)
    s2_ambiguous = []
    taue_ambiguous = []
    rex_ambiguous = []
    entropy_vals = []
    
    for i in range(len(residues)):
        # Calculate Entropy Proxy
        s2 = local_corrected_mean[i, 1]
        entropy_vals.append(calculate_entropy(s2))
        
        # Extract corrected errors using accurate indices
        taue_err = local_corrected_err[i, 0]  # Index 0 = tauE
        s2_err   = local_corrected_err[i, 1]  # Index 1 = S2
        rex_err  = local_corrected_err[i, 2]  # Index 2 = Rex
        
        # Evaluate parameters independently against their specific thresholds
        s2_ambiguous.append(s2_err > AMBIGUITY_THRESHOLD_S2)
        taue_ambiguous.append(taue_err > AMBIGUITY_THRESHOLD_TAUE)
        rex_ambiguous.append(rex_err > AMBIGUITY_THRESHOLD_REX)

    # Convert to numpy arrays for cleaner logical operations/masking later if needed
    s2_ambiguous = np.array(s2_ambiguous)
    taue_ambiguous = np.array(taue_ambiguous)
    rex_ambiguous = np.array(rex_ambiguous)

    # 7. Save Results to CSV with expanded quality tracking
    out_csv = f"{args.out}_results.csv"
    print(f"Global tau_c: {tauc_mean:.2f} ± {tauc_std:.2f} ns")
    
    with open(out_csv, 'w') as f:
        # Header updated to track parameter-specific quality
        f.write("Residue,S2,S2_err,S2_Amb,tauE_ps,tauE_err_ps,tauE_Amb,Rex,Rex_err,Rex_Amb,Entropy_proxy,tauC_global\n")
        for i, r in enumerate(residues):
            f.write(f"{r},"
                    f"{local_corrected_mean[i, 1]:.4f},{local_corrected_err[i, 1]:.4f},{'Ambiguous' if s2_ambiguous[i] else 'Good'},"
                    f"{local_corrected_mean[i, 0]:.2f},{local_corrected_err[i, 0]:.2f},{'Ambiguous' if taue_ambiguous[i] else 'Good'},"
                    f"{local_corrected_mean[i, 2]:.3f},{local_corrected_err[i, 2]:.3f},{'Ambiguous' if rex_ambiguous[i] else 'Good'},"
                    f"{entropy_vals[i]:.3f},"
                    f"{tauc_mean:.3f}\n")
    print(f"Results saved to {out_csv}")
    
    # 8. Generate Dashboard with decoupled flags passed through
    generate_dashboard(residues, local_corrected_mean, local_corrected_err,
                       s2_ambiguous, taue_ambiguous, rex_ambiguous, tauc_mean, args.out)

    # 9. (Optional) Run Explainable AI Analysis
    if args.explain:
        # Note: We use the *calibrated* values for S2 and Rex to plot against attention
        att_weights = extract_attention_weights(models, input_tensor)
        if att_weights is not None:
            plot_mechanism_explanation(
                residues, 
                att_weights, 
                local_corrected_mean[:, 1], # S2
                local_corrected_mean[:, 2], # Rex
                args.out
            )

if __name__ == "__main__":
    main()

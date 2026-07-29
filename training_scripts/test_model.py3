#!/usr/bin/env python3
"""
@author: MP Latham and Gemini
@date: Jan 15, 2026 (Updated March 2026 for dynamic field arguments)

Copied from test_global_tauc_variable_length.py3
Script to test global tau_c models trained on variable-length data.
Data generation using direct injection of knowledge about protein
systems and sparse Rex

Copied test_global_tauc_variable_length_v2.py3
Matches data generation in newest version of 
make_stratified training_data_global_tauC.py3
    -tauE is correlated with S2; rigid residues have lower tauE
     possibilities
    -Only simulating having Rex here
    -Updated noise generator function
"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import tensorflow as tf

from scipy.stats import pearsonr
from tensorflow import keras

from relaxation_rates import orig_sp_den, longitudinal_relaxation_rate_total, \
        transverse_relaxation_rate_total, nuclear_overhauser_effect

rng = np.random.default_rng()

# --- Physics Constants Helper ---
def get_constants(field_MHz):
    """Calculates field-dependent physical constants."""
    OmeH = (2 * const.pi) * (field_MHz * 10**6)
    gyroP = 267.522 * 10**6
    gyroN = -27.126 * 10**6
    OmeN = OmeH * (abs(gyroN) / gyroP)
    
    rNH = 1.02 * 10**-10  # meters
    csa_N_ppm = -160.0
    
    d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
    c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0
    
    return OmeH, OmeN, d_const_sq, c_const_sq

# --- Functions ---
def add_noise(x, scale=0.04, min_noise=0.02):
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def load_ensemble_models(model_dir):
    models = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".keras"):
            path = os.path.join(model_dir, fname)
            # When loading a model with custom objects, you might need a custom_objects dict
            # However, since GradientAccumulationModel is not saved, this should work.
            models.append(keras.models.load_model(path, compile=False))
    return models

# UPDATED DATA GENERATION FUNCTION
def generate_synthetic_data(tau_c_ns, n_residues=50, noise_level=0.04, field_MHz=850.0):
    """
    Generates a single protein of variable length for testing.
    Updated to match v7 training data biophysics (TauS/S2 correlation).
    """
    tauC = tau_c_ns * 1e-9  # seconds

    # 1. Generate S2 (Order Parameter)
    # Skewed towards 1.0 to match training distribution
    S2s = 1 - (rng.random(n_residues) ** 2)
    
    # S2f is unused in orig_sp_den but required as argument
    S2f = rng.random(n_residues)

    # 2. Generate Biophysical TauS (Internal Motion)
    # Logic: Rigid residues (S2 > 0.85) have fast motion.
    #        Flexible residues can have slow motion.
    tauE = np.zeros(n_residues)
    for i in range(n_residues):
        if S2s[i] > 0.85:
            # Rigid: Fast motion only (0-50ps)
            tauE[i] = rng.uniform(0, 0.050e-9)
        else:
            # Flexible: Can be slow (0-600ps)
            tauE[i] = rng.uniform(0, 0.600e-9)

    # Physical Constraint: tauE must be faster than global tumbling
    # Cap tauE at 40% of tauC
    tauE = np.minimum(tauE, 0.4 * tauC)

    # 3. Generate Rex (Sparse)
    # Match the "Active" training class (10-25% coverage)
    Rex = np.zeros(n_residues)
    num_exchange_residues = int(n_residues * rng.uniform(0.10, 0.25))
    
    if num_exchange_residues > 0:
        exchange_indices = rng.choice(n_residues, num_exchange_residues, replace=False)
        Rex[exchange_indices] = rng.uniform(2, 25, num_exchange_residues)

    # 4. Get Field-Dependent Constants
    OmeH, OmeN, d_const_sq, c_const_sq = get_constants(field_MHz)

    # Calculate relaxation data
    R1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH, OmeN, d_const_sq, c_const_sq)
    R2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex, OmeH, OmeN, d_const_sq, c_const_sq)
    NOE = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH, OmeN, d_const_sq, c_const_sq)

    # Add noise
    R1_noisy = add_noise(R1, scale=noise_level)
    R2_noisy = add_noise(R2, scale=noise_level)
    NOE_noisy = add_noise(NOE, scale=noise_level)

    # x shape needs to be (1, n_residues, 3) for the model input
    x = np.stack([R1_noisy, R2_noisy, NOE_noisy], axis=-1)[np.newaxis, :, :]
    
    # y_local shape is (n_residues, 3)
    y_local_true = np.stack([tauE * 1e9, S2s, Rex], axis=-1)

    return x, y_local_true, tau_c_ns

def predict_ensemble(models, x):
    global_preds = []
    local_preds = []

    for model in models:
        # model.predict expects a batch, our x is already in the correct batch format (1, N, 3)
        output = model.predict(x, verbose=0)
        local_pred, global_pred = output[0], output[1]
        global_preds.append(global_pred)
        local_preds.append(local_pred)

    # Stack predictions for analysis
    global_preds = np.vstack(global_preds) # Shape: (n_models, 1)
    local_preds = np.vstack(local_preds)   # Shape: (n_models, n_residues, 3)

    return global_preds, local_preds

def analyze_predictions(global_preds, local_preds, y_local_true, tau_c_true, output_dir):
    # Squeeze to remove unnecessary dimensions for easier analysis
    global_preds = global_preds.squeeze() # Shape: (n_models,)
    local_preds = local_preds.squeeze()   # Shape: (n_models, n_residues, 3)

    global_mean = np.mean(global_preds)
    global_std = np.std(global_preds)
    global_mae = np.abs(global_preds - tau_c_true).mean()

    print(f"\nGlobal tau_c = {tau_c_true:.2f} ns")
    print(f"  Predicted tau_c mean ± std = {global_mean:.2f} ± {global_std:.2f} ns")
    print(f"  MAE = {global_mae:.2f} ns")

    # Average the predictions across the ensemble for local parameters
    local_mean = np.mean(local_preds, axis=0) # Shape: (n_residues, 3)

    print("\n--- Pearson Correlations (Pred vs True) ---")
    for i, name in enumerate(["tau_E (ns)", "S²", "R_ex"]):
        r, p = pearsonr(local_mean[:, i], y_local_true[:, i])
        print(f"{name:>10}: r = {r:.3f}, p = {p:.2e}")

    # Plotting logic
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.hist(global_preds, bins=10, color='skyblue', edgecolor='k')
        plt.title("Predicted Global tau_c")
        plt.xlabel("tau_c (ns)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "tau_c_histogram.png"))
        plt.close()

        for i, label in enumerate(["tauE (ns)", "S2", "Rex"]):
            plt.scatter(y_local_true[:, i], local_mean[:, i], alpha=0.7)
            plt.plot([y_local_true[:, i].min(), y_local_true[:, i].max()],
                     [y_local_true[:, i].min(), y_local_true[:, i].max()], 'k--')
            plt.xlabel(f"True {label}")
            plt.ylabel(f"Predicted {label}")
            plt.title(f"Predicted vs True {label}")
            plt.savefig(os.path.join(output_dir, f"pred_vs_true_{label}.png"))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble model on synthetic variable-length test data")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory of trained Keras models")
    parser.add_argument("-tau_c", type=float, required=True, help="True global tau_c for the test protein (in ns)")
    parser.add_argument("-n_residues", type=int, default=50, help="Number of residues for the test protein")
    parser.add_argument("-noise", type=float, default=0.04, help="Noise level to add to relaxation data")
    parser.add_argument("-field", type=float, default=850.0, help="Proton resonance frequency in MHz (e.g., 850.0)")
    parser.add_argument("-output_dir", type=str, default=None, help="Directory to save output plots")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    print("Loading ensemble models...")
    models = load_ensemble_models(args.model_dir)

    print(f"Generating synthetic test data for a protein with {args.n_residues} residues at {args.field} MHz...")
    x_test, y_local_true, tau_c_true = generate_synthetic_data(
        args.tau_c, args.n_residues, args.noise, args.field)

    print("Running predictions...")
    global_preds, local_preds = predict_ensemble(models, x_test)

    analyze_predictions(global_preds, local_preds, y_local_true, tau_c_true, args.output_dir)


if __name__ == "__main__":
    main()

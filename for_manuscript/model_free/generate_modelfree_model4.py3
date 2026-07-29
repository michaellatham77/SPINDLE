#!/usr/bin/env python3
"""
Script to generate Step 2 (Model 4) Stress Test Data.
Based on 'generate_final_exam_v2.py3' physics engine.

Configuration:
- 1000 Proteins
- Max 300 Residues (to avoid Modelfree Error 9)
- Model 4 Dynamics: S2, tauE, and significant Rex

Modified on May 1st, 2026 by Olivia Krise
-Changed how S2's generated so reach full range and like in Model2
-Changed how TauE's generated so reach full range and like in Model2
-Changed the range of Rex to reach 10 s^-1
-Old script still there, just commented out
"""

import os, sys
import argparse
import numpy as np
import scipy.constants as const

parser = argparse.ArgumentParser(description="Generate stratified training data for NMR relaxation.")
parser.add_argument('-field', type=int, default=800, help='Proton resonance frequency in MHz (e.g., 850)')
parser.add_argument('-out_dir', type=str, default='800', help='Output directory')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

# --- Constants (Same as original) ---
gyroP = 267.522 * 10**6
gyroN = -27.126 * 10**6
OmeH = (2 * const.pi) * (args.field * 10**6)
OmeN = OmeH * (abs(gyroN) / gyroP)
rNH = 1.02 * 10**-10
csa_N_ppm = -160.0

# Dipole-dipole & CSA constants
d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0

rng = np.random.default_rng()

def add_noise(x, scale=0.04, min_noise=0.02):
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def orig_sp_den(freq, tauC, S2s, S2f, tauE):
    """Spectral density function (LS)"""
    tauE_arr = np.broadcast_to(tauE, S2s.shape)
    tauC_arr = np.broadcast_to(tauC, S2s.shape)
    denominator = tauC_arr + tauE_arr
    tauPr = np.divide(tauC_arr * tauE_arr, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    term1_num = S2s * tauC_arr
    term1_den = 1 + (freq * tauC_arr)**2
    term1 = np.divide(term1_num, term1_den, out=np.zeros_like(term1_den), where=term1_den!=0)
    
    term2_num = (1 - S2s) * tauPr
    term2_den = 1 + (freq * tauPr)**2
    term2 = np.divide(term2_num, term2_den, out=np.zeros_like(term2_den), where=term2_den!=0)
    
    return (term1 + term2) * (2.0/5.0)

def calculate_rates(tauC, S2s, tauE, Rex, S2f):
    """Calculates R1, R2, NOE using the physics functions"""
   # S2f = np.ones_like(S2s) # Not used in simple LS, but required by function signature
    J_func = orig_sp_den

    # R1
    J_DD_R1 = (J_func(OmeH - OmeN, tauC, S2s, S2f, tauE) +
               3 * J_func(OmeN, tauC, S2s, S2f, tauE) +
               6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE))
    R1_DD = (d_const_sq / 4.0) * J_DD_R1
    R1_CSA = (c_const_sq) * J_func(OmeN, tauC, S2s, S2f, tauE)
    R1 = R1_DD + R1_CSA

    # R2
    J_DD_R2 = (4 * J_func(0, tauC, S2s, S2f, tauE) +
               J_func(OmeH - OmeN, tauC, S2s, S2f, tauE) +
               3 * J_func(OmeN, tauC, S2s, S2f, tauE) +
               6 * J_func(OmeH, tauC, S2s, S2f, tauE) +
               6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE))
    R2_DD = (d_const_sq / 8.0) * J_DD_R2
    
    J_CSA_R2 = (4 * J_func(0, tauC, S2s, S2f, tauE) +
                3 * J_func(OmeN, tauC, S2s, S2f, tauE))
    R2_CSA = (c_const_sq / 6.0) * J_CSA_R2
    R2 = R2_DD + R2_CSA + Rex

    # NOE
    sigma_NH = ((d_const_sq / 4.0) *
                (6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE) -
                 J_func(OmeH - OmeN, tauC, S2s, S2f, tauE)))
    NOE = 1 + (gyroP / gyroN) * (1/R1) * sigma_NH
    
    return R1, R2, NOE

# --- 2. Data Generation Logic ---

def generate_step2_protein(n_residues, tau_c_ns):
    tauC = tau_c_ns * 1e-9

    # -- PARAMETER DISTRIBUTIONS (Step 2 Stress Test) --
    
    #Generate S2 (Skewed towards 1.0)
    S2s = 1 - (rng.random(n_residues) **2)
    S2f = rng.random(n_residues)

# S2: 0.50 - 0.95
    #S2s = rng.uniform(0.50, 0.95, n_residues)
    

# 2. Generate Biophysical tauE
    tauE = np.zeros(n_residues)
    for i in range(n_residues):
        if S2s[i] > 0.80:
            tauE[i] = rng.uniform(0, 0.050e-9)
        else:
            tauE[i] = rng.uniform(0, 1.0e-9)
    tauE = np.minimum(tauE, 0.4 * tauC)


    # tauE: 10 - 500 ps (Mixed into ns)
   # tauE_ps = rng.uniform(10.0, 500.0, n_residues)
    #tauE = tauE_ps / 1000.0 * 1e-9 # Convert ps -> ns -> seconds
    
    # Rex: 30% of residues, 0.5 - 10.0 s^-1
    Rex = np.zeros(n_residues)
    mask_exchange = rng.random(n_residues) < 0.30  # 30% Probability
    Rex[mask_exchange] = rng.uniform(0.5, 10.0, np.sum(mask_exchange))

    # Calculate Physics
    R1, R2, NOE = calculate_rates(tauC, S2s, tauE, Rex, S2f)

    # Add Noise
    R1_noisy = add_noise(R1)
    R2_noisy = add_noise(R2)
    NOE_noisy = add_noise(NOE)

    # Errors (4%)
    R1_err = 0.04 * R1_noisy
    R2_err = 0.04 * R2_noisy
    NOE_err = 0.04 * np.abs(NOE_noisy)

    # Pack Data for .npz
    features = np.stack([R1_noisy, R2_noisy, NOE_noisy], axis=-1).astype(np.float32)
    
    labels = {
        "tauC_ns": tau_c_ns,
        "S2": S2s.astype(np.float32),
        "tauE_ns": (tauE * 1e9).astype(np.float32), 
        "Rex": Rex.astype(np.float32)
    }

    # Pack Data for .txt
    txt_data = (R1_noisy, R1_err, R2_noisy, R2_err, NOE_noisy, NOE_err)

    return features, labels, txt_data

# --- 3. Main Execution ---
os.makedirs(args.out_dir, exist_ok=True)

N_PROTEINS = 1000
OUTPUT_NPZ = os.path.join(args.out_dir, "modelfree_model4_dataset.npz")
OUTPUT_TXT = os.path.join(args.out_dir, "modelfree_model4_benchmark_data.txt")

print(f"Generating {N_PROTEINS} proteins (Model 4 Stress Test)...")
print(f"Constraints: Max 300 Residues. Rex Probability 30%.")

all_features = []
all_labels = []

with open(OUTPUT_TXT, "w") as mf_file:
    for i in range(N_PROTEINS):
        # Size: 40 - 300 (Strict limit for Modelfree)
        n_res = rng.integers(40, 301)
        
        # tauC: 3 - 15 ns
        tau_c_ns = rng.uniform(3.0, 15.0)
        
        features, labels, txt_data = generate_step2_protein(n_res, tau_c_ns)
        
        all_features.append(features)
        all_labels.append(labels)
        
        # Write Text File (for make_mfdata pipeline)
        r1, r1e, r2, r2e, noe, noee = txt_data
        for j in range(n_res):
            mf_file.write(f"{j+1:<4d} {r1[j]:.4f} {r1e[j]:.4f} {r2[j]:.4f} {r2e[j]:.4f} {noe[j]:.4f} {noee[j]:.4f}\n")
        mf_file.write("END\n")

# Save NPZ (for ML Benchmark)
np.savez_compressed(
    OUTPUT_NPZ,
    features=np.array(all_features, dtype=object),
    labels=np.array(all_labels, dtype=object)
)

print(f"\n[Done]")
print(f"1. Ground Truth saved to: {OUTPUT_NPZ}")
print(f"2. Modelfree Inputs saved to: {OUTPUT_TXT}")

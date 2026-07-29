#!/usr/bin/env python3
"""
Script to make testing data for modelfree.
Strictly S2 and tauE (i.e., model 2). No Rex.

 @author MP Latham
 @date Jan 23 2026
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

d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0

rng = np.random.default_rng()

def add_noise(x, scale=0.04, min_noise=0.02):
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def orig_sp_den(freq, tauC, S2s, S2f, tauE):
    # Standard Spectral Density
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

def longitudinal_relaxation_rate_total(J_func, tauC, S2s, S2f, tauE):
    J_DD_terms = (J_func(OmeH - OmeN, tauC, S2s, S2f, tauE) +
                  3 * J_func(OmeN, tauC, S2s, S2f, tauE) +
                  6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE))
    R1_DD = (d_const_sq / 4.0) * J_DD_terms
    R1_CSA = (c_const_sq) * J_func(OmeN, tauC, S2s, S2f, tauE)
    return R1_DD + R1_CSA

def transverse_relaxation_rate_total(J_func, tauC, S2s, S2f, tauE, Rex):
    J_DD_terms = (4 * J_func(0, tauC, S2s, S2f, tauE) +
                  J_func(OmeH - OmeN, tauC, S2s, S2f, tauE) +
                  3 * J_func(OmeN, tauC, S2s, S2f, tauE) +
                  6 * J_func(OmeH, tauC, S2s, S2f, tauE) +
                  6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE))
    R2_DD = (d_const_sq / 8.0) * J_DD_terms
    J_CSA_terms = (4 * J_func(0, tauC, S2s, S2f, tauE) +
                   3 * J_func(OmeN, tauC, S2s, S2f, tauE))
    R2_CSA = (c_const_sq / 6.0) * J_CSA_terms
    return R2_DD + R2_CSA + Rex

def nuclear_overhauser_effect(J_func, tauC, S2s, S2f, tauE):
    r1 = longitudinal_relaxation_rate_total(J_func, tauC, S2s, S2f, tauE)
    sigma_NH = ((d_const_sq / 4.0) *
    (6 * J_func(OmeH + OmeN, tauC, S2s, S2f, tauE) -
    J_func(OmeH - OmeN, tauC, S2s, S2f, tauE)))
    return 1 + (gyroP / gyroN) * (1/r1) * sigma_NH

def generate_protein_data(n_residues, tau_c_ns):
    tauC = tau_c_ns * 1e-9

    # 1. Generate S2 (Skewed towards 1.0)
    S2s = 1 - (rng.random(n_residues) ** 2)
    S2f = rng.random(n_residues) 

    # 2. Generate Biophysical tauE
    tauE = np.zeros(n_residues)
    for i in range(n_residues):
        if S2s[i] > 0.80:
            tauE[i] = rng.uniform(0, 0.050e-9)
        else:
            tauE[i] = rng.uniform(0, 1.0e-9)
    tauE = np.minimum(tauE, 0.4 * tauC)

    # Force Rex to be 0 for ALL residues (Model 2 only)
    # The random injection of exchange residues  is REMOVED.
    Rex = np.zeros(n_residues) 

    # Calculate Rates
    R1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE)
    R2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex)
    NOE = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE)

    # Add Noise
    R1_noisy = add_noise(R1)
    R2_noisy = add_noise(R2)
    NOE_noisy = add_noise(NOE)

    # Error estimates
    R1_err = 0.04 * R1_noisy
    R2_err = 0.04 * R2_noisy
    NOE_err = 0.04 * np.abs(NOE_noisy)

    features = np.stack([R1_noisy, R2_noisy, NOE_noisy], axis=-1).astype(np.float32)

    labels = {
        "tauC_ns": tau_c_ns,
        "S2": S2s.astype(np.float32),
        "tauE_ns": (tauE * 1e9).astype(np.float32),
        "Rex": Rex.astype(np.float32)
    }

    modelfree_data = (R1_noisy, R1_err, R2_noisy, R2_err, NOE_noisy, NOE_err)

    return features, labels, modelfree_data


# --- Main Logic ---
os.makedirs(args.out_dir, exist_ok=True)

N_PROTEINS_TO_GENERATE = 1000 
rng = np.random.default_rng() 

all_features = []
all_labels = []
# Updated filename for Step 1
modelfree_file = open(os.path.join(args.out_dir, "modelfree_model2_benchmark_data.txt"), "w")

print(f"Generating {N_PROTEINS_TO_GENERATE} proteins (Model 2 Only - No Rex)...")
for i in range(N_PROTEINS_TO_GENERATE):
    n_residues = rng.integers(40, 426)
    base_tauC_ns = 0.0005998 * n_residues * 110 + 0.1674
    tau_c_ns = base_tauC_ns * rng.uniform(0.5, 1.5)
    
    features, labels, modelfree_data = generate_protein_data(n_residues, tau_c_ns)
    
    all_features.append(features)
    all_labels.append(labels)
    
    r1, r1e, r2, r2e, noe, noee = modelfree_data
    for j in range(n_residues):
        modelfree_file.write(f"{j+1:<4d} {r1[j]:.4f} {r1e[j]:.4f} {r2[j]:.4f} {r2e[j]:.4f} {noe[j]:.4f} {noee[j]:.4f}\n")
    modelfree_file.write("END\n")

modelfree_file.close()

npz_filename = os.path.join(args.out_dir, 'modelfree_model2_dataset.npz')
np.savez_compressed(
    npz_filename,
    features=np.array(all_features, dtype=object),
    labels=np.array(all_labels, dtype=object)
)
print("\n'modelfree_model2_dataset.npz' and 'modelfree_model2_benchmark_data.txt' created.")

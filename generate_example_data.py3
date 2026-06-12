#!/usr/bin/env python3
"""
Generate a synthetic example dataset for testing SPINDLE.
Outputs a standard text file containing space-separated R1, R2, and NOE values,
along with a companion ground-truth file for accuracy validation.

Adapted from the master SPINDLE data generation pipeline.
"""

import sys, os
import argparse
import numpy as np
import scipy.constants as const

# Import the relaxation rate engine from your codebase
try:
    from relaxation_rates import orig_sp_den, longitudinal_relaxation_rate_total, \
        transverse_relaxation_rate_total, nuclear_overhauser_effect
except ImportError:
    print("Error: 'relaxation_rates.py' must be in the same directory or Python path.")
    sys.exit(1)

def add_noise(x, scale=0.04, min_noise=0.02):
    """Adds realistic experimental noise using a proportional scale and noise floor."""
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def main():
    parser = argparse.ArgumentParser(description="Generate a single-protein synthetic dataset for SPINDLE testing.")
    parser.add_argument('-field', type=int, default=600, help='Proton resonance frequency in MHz (e.g., 600, 800)')
    parser.add_argument('-n_residues', type=int, default=76, help='Number of residues to simulate (default: 76, e.g., Ubiquitin)')
    # FIXED: Escaped the percent character by changing '4%' to '4%%'
    parser.add_argument('-noise_scale', type=float, default=0.04, help='Proportional noise level factor (default: 0.04 for 4%%)')
    parser.add_argument('-out_prefix', type=str, default='synthetic_protein', help='Prefix for output filenames')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    rng = np.random.default_rng()

    # --- Physics Constants Setup ---
    OmeH = (2 * const.pi) * (args.field * 10**6)
    gyroP = 267.522 * 10**6  # rad s-1 T-1
    gyroN = -27.126 * 10**6   # rad s-1 T-1
    OmeN = OmeH * (abs(gyroN) / gyroP) 
    
    rNH = 1.02 * 10**-10  # meters
    csa_N_ppm = -160.0

    d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
    c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0

    # --- Parameter Generation (Matching SPINDLE Validation Distributions) ---
    n_res = args.n_residues
    residues = np.arange(1, n_res + 1)

    # 1. Generate global tumbling tauC (ns)
    base_tauC_ns = 0.0005998 * n_res * 110 + 0.1674
    randomized_tauC_ns = base_tauC_ns * rng.uniform(0.8, 1.2) # Realistic localized scaling window
    tauC = randomized_tauC_ns * 1e-9
    
    # 2. Generate Order Parameters (S2)
    #S2s = 1.0 - (rng.random(n_res) ** 2) 
    S2s = 0.85 * np.ones(n_res)
    S2f = rng.random(n_res)

    # 3. Generate Local Internal Timescales (tauE)
    tauE = np.zeros(n_res)
    for i in range(n_res):
        if S2s[i] > 0.80:
            tauE[i] = rng.uniform(0, 0.050e-9)
        else:
            tauE[i] = rng.uniform(0, 1.0e-9)
    tauE = np.minimum(tauE, 0.4 * tauC) # Bound physically against global tumbling clock

    # 4. Generate Chemical Exchange Rates (Rex)
    Rex = np.zeros(n_res)
    # Force ~20% of residues to experience active chemical exchange events
    #num_exchange_residues = max(1, int(n_res * rng.uniform(0.15, 0.25)))
    #exchange_indices = rng.choice(n_res, num_exchange_residues, replace=False)
    #Rex[exchange_indices] = rng.uniform(2.0, 15.0, num_exchange_residues)

    # --- Calculate True Relaxation Rates ---
    R1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE,
                                            OmeH, OmeN, d_const_sq, c_const_sq)
    R2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex,
                                           OmeH, OmeN, d_const_sq, c_const_sq)
    NOE = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE,
                                    OmeH, OmeN, d_const_sq, c_const_sq)

    # --- Superimpose Experimental Noise Floor ---
    R1_noisy = add_noise(R1, scale=args.noise_scale)
    R2_noisy = add_noise(R2, scale=args.noise_scale)
    NOE_noisy = add_noise(NOE, scale=args.noise_scale)

    # --- Export File 1: Space-Separated Experimental Data Format ---
    exp_filename = f"{args.out_prefix}_{args.field}MHz_data.txt"
    with open(exp_filename, 'w') as f:
        f.write(f"# SPINDLE Synthetic Benchmark Dataset - Generated Field: {args.field} MHz\n")
        f.write(f"# Estimated Global tau_c: {randomized_tauC_ns:.3f} ns\n")
        f.write(f"# Residue   R1          R2          NOE\n")
        for i in range(n_res):
            f.write(f"{residues[i]:-10d}  {R1_noisy[i]:-10.4f}  {R2_noisy[i]:-10.4f}  {NOE_noisy[i]:-10.4f}\n")

    # --- Export File 2: Ground Truth Benchmarking Matrix ---
    truth_filename = f"{args.out_prefix}_{args.field}MHz_truth.csv"
    with open(truth_filename, 'w') as f:
        f.write("Residue,True_S2,True_tauE_ps,True_Rex_s1,True_tauC_ns\n")
        for i in range(n_res):
            f.write(f"{residues[i]},"
                    f"{S2s[i]:.4f},"
                    f"{tauE[i]*1e12:.2f},"  # Exported in picoseconds (ps) for user clarity
                    f"{Rex[i]:.3f},"
                    f"{randomized_tauC_ns:.3f}\n")

    print(f"\nExample generation successful!")
    print(f" -> Simulated Input Data (Upload to SPINDLE): {exp_filename}")
    print(f" -> Benchmarking Reference Matrix:            {truth_filename}")

if __name__ == "__main__":
    main()

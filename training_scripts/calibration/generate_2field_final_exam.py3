#!/usr/bin/env python3
"""
Script to make final 2-field testing data (600 MHz & 800 MHz)

Updated script to follow new logic for 2-field data generation
Refactored to use argparse and unified relaxation_rates library

@author: MP Latham
@date: Jan. 17, 2026 (Updated for 2-field March 2026)
"""

import sys, os
import argparse
import numpy as np
import scipy.constants as const

from relaxation_rates import orig_sp_den, longitudinal_relaxation_rate_total, \
    transverse_relaxation_rate_total, nuclear_overhauser_effect

rng = np.random.default_rng()

# Constants Helper
def get_constants(field_MHz):
    """Calculates field-dependent physical constants."""
    OmeH = (2 * const.pi) * (field_MHz * 10**6)
    gyroP = 267.522 * 10**6 #rad s-1 T-1
    gyroN = -27.126 * 10**6
    OmeN = OmeH * (abs(gyroN) / gyroP)
    
    rNH = 1.02 * 10**-10  # meters
    csa_N_ppm = -160.0
    
    d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
    c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0
    
    return OmeH, OmeN, d_const_sq, c_const_sq

def add_noise(x, scale=0.04, min_noise=0.02):
    # min_noise represents absolute error floor
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def generate_protein_data(n_residues, tau_c_ns, field1_MHz=600.0, field2_MHz=800.0):
    tauC = tau_c_ns * 1e-9

    # 1. Generate S2 (Skewed towards 1.0)
    S2s = 1 - (rng.random(n_residues) ** 2)
    S2f = rng.random(n_residues) # Unused but required for function call

    # 2. Generate tauE (Continuous Scaling)
    tauE = np.zeros(n_residues)
    for i in range(n_residues):
        # Smoothly scales max tauE from 1.0 ns (at S2=0) down to 0.3 ns (at S2=1)
        max_tauE_for_residue = 1.0e-9 * (1.0 - 0.7 * S2s[i])
        tauE[i] = rng.uniform(0, max_tauE_for_residue)

    tauE = np.minimum(tauE, 0.4 * tauC)

    # 3. Generate Rex (Sparse/Mixed) relative to Field 1
    Rex_1 = np.zeros(n_residues)
    num_exchange_residues = int(n_residues * rng.uniform(0.10, 0.25))
    if num_exchange_residues > 0:
        exchange_indices = rng.choice(n_residues, num_exchange_residues, replace=False)
        Rex_1[exchange_indices] = rng.uniform(1, 25, num_exchange_residues)
        
    # Scale Rex for Field 2 (Rex scales with B0^2)
    rex_scaling_factor = (field2_MHz / field1_MHz)**2
    Rex_2 = Rex_1 * rex_scaling_factor

    # 4. Get Field-Dependent Constants
    OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1 = get_constants(field1_MHz)
    OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2 = get_constants(field2_MHz)

    # Calculate Rates - Field 1
    R1_1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)
    R2_1 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex_1, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)
    NOE_1 = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)

    # Calculate Rates - Field 2
    R1_2 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)
    R2_2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex_2, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)
    NOE_2 = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)

    # Add Noise
    R1_1_noisy, R2_1_noisy, NOE_1_noisy = add_noise(R1_1), add_noise(R2_1), add_noise(NOE_1)
    R1_2_noisy, R2_2_noisy, NOE_2_noisy = add_noise(R1_2), add_noise(R2_2), add_noise(NOE_2)

    # Error estimates (for Modelfree benchmark file)
    R1_1_err, R2_1_err, NOE_1_err = 0.04 * R1_1_noisy, 0.04 * R2_1_noisy, 0.04 * np.abs(NOE_1_noisy)
    R1_2_err, R2_2_err, NOE_2_err = 0.04 * R1_2_noisy, 0.04 * R2_2_noisy, 0.04 * np.abs(NOE_2_noisy)

    # Format Features (6 Channels)
    features = np.stack([
        R1_1_noisy, R2_1_noisy, NOE_1_noisy, 
        R1_2_noisy, R2_2_noisy, NOE_2_noisy
    ], axis=-1).astype(np.float32)

    # Format Labels (Rex saved relative to 600 MHz)
    labels = {
        "tauC_ns": tau_c_ns,
        "S2": S2s.astype(np.float32),
        "tauE_ns": (tauE * 1e9).astype(np.float32), 
        "Rex": Rex_1.astype(np.float32) 
    }

    modelfree_data = (
        R1_1_noisy, R1_1_err, R2_1_noisy, R2_1_err, NOE_1_noisy, NOE_1_err,
        R1_2_noisy, R1_2_err, R2_2_noisy, R2_2_err, NOE_2_noisy, NOE_2_err
    )

    return features, labels, modelfree_data

# Main 
def main():
    parser = argparse.ArgumentParser(description="Generate final 2-field test dataset for model evaluation")
    parser.add_argument("-n_proteins", type=int, default=10000, help="Number of proteins to generate")
    parser.add_argument("-out_dir", type=str, default=".", help="Directory to save the generated dataset files")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_features = []
    all_labels = []
    
    txt_out_path = os.path.join(args.out_dir, "modelfree_benchmark_data_2field.txt")
    npz_out_path = os.path.join(args.out_dir, "final_exam_dataset_2field.npz")
    
    print(f"Generating {args.n_proteins} diverse test proteins at 600 and 800 MHz...")
    
    with open(txt_out_path, "w") as modelfree_file:
        for i in range(args.n_proteins):
            n_residues = rng.integers(40, 426)
            base_tauC_ns = 0.0005998 * n_residues * 110 + 0.1674
            tau_c_ns = base_tauC_ns * rng.uniform(0.5, 1.5)
            
            features, labels, modelfree_data = generate_protein_data(n_residues, tau_c_ns, 600.0, 800.0)
            
            all_features.append(features)
            all_labels.append(labels)
            
            # Write 2-field data to modelfree file (12 data columns)
            r1_1, r1e_1, r2_1, r2e_1, noe_1, noee_1, r1_2, r1e_2, r2_2, r2e_2, noe_2, noee_2 = modelfree_data
            for j in range(n_residues):
                modelfree_file.write(
                    f"{j+1:<4d} "
                    f"{r1_1[j]:.4f} {r1e_1[j]:.4f} {r2_1[j]:.4f} {r2e_1[j]:.4f} {noe_1[j]:.4f} {noee_1[j]:.4f} "
                    f"{r1_2[j]:.4f} {r1e_2[j]:.4f} {r2_2[j]:.4f} {r2e_2[j]:.4f} {noe_2[j]:.4f} {noee_2[j]:.4f}\n"
                )
            modelfree_file.write("END\n")

    np.savez_compressed(
        npz_out_path,
        features=np.array(all_features, dtype=object),
        labels=np.array(all_labels, dtype=object)
    )
    print(f"\nFiles saved to '{args.out_dir}':")
    print(f"  - modelfree_benchmark_data_2field.txt")
    print(f"  - final_exam_dataset_2field.npz")

if __name__ == "__main__":
    main()

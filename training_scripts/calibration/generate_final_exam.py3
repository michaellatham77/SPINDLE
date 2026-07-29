#!/usr/bin/env python3
"""
Script to make final testing data

Updated script to follow new logic for data generation
Refactored to use argparse and unified relaxation_rates library

@author: MP Latham
@date: Jan. 17, 2026 (Updated March 2026)
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

def generate_protein_data(n_residues, tau_c_ns, field_MHz):
    tauC = tau_c_ns * 1e-9

    # 1. Generate S2 (Skewed towards 1.0)
    S2s = 1 - (rng.random(n_residues) ** 2)
    S2f = rng.random(n_residues) # Unused but required for function call

    # 2. Generate tauE (Correlated with S2)
    #tauE = np.zeros(n_residues)
    #for i in range(n_residues):
    #    if S2s[i] > 0.80:
    #        # Rigid: Fast motion (0-50ps)
    #        tauE[i] = rng.uniform(0, 0.050e-9)
    #    else:
    #        # Flexible: Up to 1.0 ns
    #        tauE[i] = rng.uniform(0, 1.0e-9)
    #
    # Safety Valve: tauE << tauC (max 40%)
    #tauE = np.minimum(tauE, 0.4 * tauC)

    # 3. Generate tauE (Continuous Scaling)
    tauE = np.zeros(n_residues)
    for i in range(n_residues):
        # Smoothly scales max tauE from 1.0 ns (at S2=0) down to 0.3 ns (at S2=1)
        max_tauE_for_residue = 1.0e-9 * (1.0 - 0.7 * S2s[i])
        tauE[i] = rng.uniform(0, max_tauE_for_residue)

    tauE = np.minimum(tauE, 0.4 * tauC)

    # 3. Generate Rex (Sparse/Mixed)
    Rex = np.zeros(n_residues)
    num_exchange_residues = int(n_residues * rng.uniform(0.10, 0.25))
    if num_exchange_residues > 0:
        exchange_indices = rng.choice(n_residues, num_exchange_residues, replace=False)
        Rex[exchange_indices] = rng.uniform(1, 25, num_exchange_residues)

    # 4. Get Field-Dependent Constants
    OmeH, OmeN, d_const_sq, c_const_sq = get_constants(field_MHz)

    # Calculate Rates
    R1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH, OmeN, d_const_sq, c_const_sq)
    R2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex, OmeH, OmeN, d_const_sq, c_const_sq)
    NOE = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH, OmeN, d_const_sq, c_const_sq)

    # Add Noise
    R1_noisy = add_noise(R1)
    R2_noisy = add_noise(R2)
    NOE_noisy = add_noise(NOE)

    # Error estimates (for Modelfree benchmark file)
    R1_err = 0.04 * R1_noisy
    R2_err = 0.04 * R2_noisy
    NOE_err = 0.04 * np.abs(NOE_noisy)

    # Format Features
    features = np.stack([R1_noisy, R2_noisy, NOE_noisy], axis=-1).astype(np.float32)

    # Format Labels
    labels = {
        "tauC_ns": tau_c_ns,
        "S2": S2s.astype(np.float32),
        "tauE_ns": (tauE * 1e9).astype(np.float32), 
        "Rex": Rex.astype(np.float32)
    }

    modelfree_data = (R1_noisy, R1_err, R2_noisy, R2_err, NOE_noisy, NOE_err)

    return features, labels, modelfree_data


# Main 
def main():
    parser = argparse.ArgumentParser(description="Generate final exam test dataset for model evaluation")
    parser.add_argument("-field", type=float, default=850.0, help="Proton resonance frequency in MHz (e.g., 850.0)")
    parser.add_argument("-n_proteins", type=int, default=10000, help="Number of proteins to generate")
    parser.add_argument("-out_dir", type=str, default=".", help="Directory to save the generated dataset files")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    all_features = []
    all_labels = []
    
    txt_out_path = os.path.join(args.out_dir, "modelfree_benchmark_data.txt")
    npz_out_path = os.path.join(args.out_dir, "final_exam_dataset.npz")
    
    print(f"Generating {args.n_proteins} diverse test proteins at {args.field} MHz...")
    
    with open(txt_out_path, "w") as modelfree_file:
        for i in range(args.n_proteins):
            # Generate a diverse range of protein sizes and tumbling times
            n_residues = rng.integers(40, 426)
            base_tauC_ns = 0.0005998 * n_residues * 110 + 0.1674
            tau_c_ns = base_tauC_ns * rng.uniform(0.5, 1.5)
            
            features, labels, modelfree_data = generate_protein_data(n_residues, tau_c_ns, args.field)
            
            all_features.append(features)
            all_labels.append(labels)
            
            # Write to modelfree file
            r1, r1e, r2, r2e, noe, noee = modelfree_data
            for j in range(n_residues):
                modelfree_file.write(f"{j+1:<4d} {r1[j]:.4f} {r1e[j]:.4f} {r2[j]:.4f} {r2e[j]:.4f} {noe[j]:.4f} {noee[j]:.4f}\n")
            modelfree_file.write("END\n") # Separator for different proteins

    # Save the dataset for our model
    np.savez_compressed(
        npz_out_path,
        features=np.array(all_features, dtype=object),
        labels=np.array(all_labels, dtype=object)
    )
    print(f"\nFiles saved to '{args.out_dir}':")
    print(f"  - modelfree_benchmark_data.txt")
    print(f"  - final_exam_dataset.npz")

if __name__ == "__main__":
    main()

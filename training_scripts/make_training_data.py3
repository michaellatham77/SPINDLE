#!/usr/bin/env python3
"""
Script to generate training data using global tauC
and variable length number of residues

@author: MP Latham
@data: Jan. 14, 2026
"""

import sys, os
import argparse
import tensorflow as tf
import numpy as np
import scipy.constants as const

from relaxation_rates import orig_sp_den, longitudinal_relaxation_rate_total, \
    transverse_relaxation_rate_total, nuclear_overhauser_effect

# TFRecord Helper Functions 
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(features, local_labels, global_label):
    feature = {
        'features': _bytes_feature(tf.io.serialize_tensor(features)),
        'local_labels': _bytes_feature(tf.io.serialize_tensor(local_labels)),
        'global_label': _float_feature(global_label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def add_noise(x, scale=0.04, min_noise=0.02):
    # min_noise represents absolute error floor
    sigma = np.sqrt((scale * x)**2 + min_noise**2)
    return x + np.random.normal(0, sigma)

def main():
    # --- Set up argparse ---
    parser = argparse.ArgumentParser(description="Generate stratified training data for NMR relaxation.")
    parser.add_argument('-field', type=int, default=850, help='Proton resonance frequency in MHz (e.g., 850)')
    parser.add_argument('-n_proteins', type=int, default=400000, help='Total number of proteins to generate')
    parser.add_argument('-out_dir', type=str, default='850', help='Output directory for TFRecords')
    parser.add_argument('-n_files', type=int, default=40, help='Number of TFRecord files to split the data into')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()

    # Physics Constants
    # Calculated based on the -field argument
    OmeH = (2 * const.pi) * (args.field * 10**6)
    gyroP = 267.522 * 10**6  # rad s-1 T-1
    gyroN = -27.126 * 10**6
    
    # Calculate 15N frequency based on the gyromagnetic ratio
    OmeN = OmeH * (abs(gyroN) / gyroP) 
    
    rNH = 1.02 * 10**-10  # meters
    csa_N_ppm = -160.0

    # Dipole-dipole constant squared (d^2)
    d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2

    # CSA constant squared (c^2) 
    c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0

    # Main Data Generation
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Don't use GPUs for data generation
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng()

    # Stratified Sampling
    n_bins = 4
    proteins_per_bin = args.n_proteins // n_bins
    proteins_per_bin_per_file = proteins_per_bin // args.n_files

    residue_bins = [
        (40, 140),    # Bin 1: Small proteins
        (141, 240),   # Bin 2: Medium proteins
        (241, 340),   # Bin 3: Large proteins
        (341, 451)    # Bin 4: Very large proteins
    ]

    for file_idx in range(args.n_files):
        output_filename = os.path.join(args.out_dir, f'training_data-{file_idx:03d}-of-{args.n_files:03d}.tfrecord')
        with tf.io.TFRecordWriter(output_filename) as writer:
            print(f"Writing to {output_filename}...")
            
            for min_res, max_res in residue_bins:
                print(f"  Generating proteins in range {min_res}-{max_res}...")
                for _ in range(proteins_per_bin_per_file):
                    
                    noise_level = rng.uniform(0.02, 0.08)
                    n_residues = rng.integers(min_res, max_res + 1)
                    
                    # 1. Generate global tauC
                    base_tauC_ns = 0.0005998 * n_residues * 110 + 0.1674
                    randomized_tauC_ns = base_tauC_ns * rng.uniform(0.5, 1.5)
                    tauC = randomized_tauC_ns * 1e-9
                    
                    # 2. Generate S2
                    S2s = 1 - (rng.random(n_residues) ** 2) 
                    S2f = rng.random(n_residues)

                    # 3. Generate tauE
                    #tauE = np.zeros(n_residues)
                    #for i in range(n_residues):
                    #    if S2s[i] > 0.80:
                    #        tauE[i] = rng.uniform(0, 0.050e-9)
                    #    else:
                    #        tauE[i] = rng.uniform(0, 1.0e-9)

                    #tauE = np.minimum(tauE, 0.4 * tauC)

                    # 3. Generate tauE (Continuous Scaling)
                    tauE = np.zeros(n_residues)
                    for i in range(n_residues):
                        # Smoothly scales max tauE from 1.0 ns (at S2=0) down to 0.3 ns (at S2=1)
                        max_tauE_for_residue = 1.0e-9 * (1.0 - 0.7 * S2s[i])
                        tauE[i] = rng.uniform(0, max_tauE_for_residue)

                    tauE = np.minimum(tauE, 0.4 * tauC)

                    # 4. Generate Rex
                    Rex = np.zeros(n_residues)
                    if rng.random() <= 0.5:
                        num_exchange_residues = int(n_residues * rng.uniform(0.10, 0.25))
                        if num_exchange_residues > 0:
                            exchange_indices = rng.choice(n_residues, num_exchange_residues, replace=False)
                            Rex[exchange_indices] = rng.uniform(1.0, 25.0, num_exchange_residues)

                    # Calculate relaxation data
                    R1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE,
                            OmeH, OmeN, d_const_sq, c_const_sq)
                    R2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex,
                            OmeH, OmeN, d_const_sq, c_const_sq)
                    NOE = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE,
                            OmeH, OmeN, d_const_sq, c_const_sq)

                    # Add noise
                    R1_noisy = add_noise(R1, scale=noise_level)
                    R2_noisy = add_noise(R2, scale=noise_level)
                    NOE_noisy = add_noise(NOE, scale=noise_level)
                    
                    # Format and serialize
                    features_tensor = np.stack([R1_noisy, R2_noisy, NOE_noisy], axis=-1).astype(np.float32)
                    local_labels_tensor = np.stack([tauE * 1e9, S2s, Rex], axis=-1).astype(np.float32)
                    global_label_val = tauC * 1e9
                    
                    serialized_example = serialize_example(features_tensor, local_labels_tensor, global_label_val)
                    writer.write(serialized_example)

    print("\nStratified data generation complete!")

if __name__ == "__main__":
    main()

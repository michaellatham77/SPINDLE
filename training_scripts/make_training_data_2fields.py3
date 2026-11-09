#!/usr/bin/env python3
"""
Script to generate 2-field training data using global tauC
and variable length number of residues (600 MHz and 800 MHz)
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

def calculate_physics_constants(field_mhz):
    OmeH = (2 * const.pi) * (field_mhz * 10**6)
    gyroP = 267.522 * 10**6
    gyroN = -27.126 * 10**6
    OmeN = OmeH * (abs(gyroN) / gyroP) 
    
    rNH = 1.02 * 10**-10
    csa_N_ppm = -160.0

    d_const_sq = (((const.mu_0 * const.h * gyroP * gyroN) / (8 * const.pi**2)) * (1 / (rNH**3)))**2
    c_const_sq = ((OmeN * csa_N_ppm * 10**-6)**2) / 3.0
    
    return OmeH, OmeN, d_const_sq, c_const_sq

def main():
    parser = argparse.ArgumentParser(description="Generate stratified two-field training data.")
    # Removed -field argument since we are hardcoding two specific fields
    parser.add_argument('-n_proteins', type=int, default=400000, help='Total number of proteins to generate')
    parser.add_argument('-out_dir', type=str, default='600_800', help='Output directory for TFRecords')
    parser.add_argument('-n_files', type=int, default=40, help='Number of TFRecord files')
    
    args = parser.parse_args()

    # Calculate constants for both fields
    field1_mhz = 600
    field2_mhz = 800
    
    OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1 = calculate_physics_constants(field1_mhz)
    OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2 = calculate_physics_constants(field2_mhz)

    # Scaling Rex for the second field (Rex scales with the square of the magnetic field)
    # If Rex is defined at 600 MHz, at 800 MHz it will be Rex * (800/600)^2
    rex_scaling_factor = (field2_mhz / field1_mhz)**2

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng()

    n_bins = 4
    proteins_per_bin = args.n_proteins // n_bins
    proteins_per_bin_per_file = proteins_per_bin // args.n_files

    residue_bins = [
        (40, 140),
        (141, 240),
        (241, 340),
        (341, 451)
    ]

    for file_idx in range(args.n_files):
        output_filename = os.path.join(args.out_dir, f'training_data-{file_idx:03d}-of-{args.n_files:03d}.tfrecord')
        with tf.io.TFRecordWriter(output_filename) as writer:
            print(f"Writing to {output_filename}...")
            
            for min_res, max_res in residue_bins:
                for _ in range(proteins_per_bin_per_file):
                    noise_level = rng.uniform(0.02, 0.08)
                    n_residues = rng.integers(min_res, max_res + 1)
                    
                    base_tauC_ns = 0.0005998 * n_residues * 110 + 0.1674
                    randomized_tauC_ns = base_tauC_ns * rng.uniform(0.5, 1.5)
                    tauC = randomized_tauC_ns * 1e-9
                    
                    S2s = 1 - (rng.random(n_residues) ** 2) 
                    S2f = rng.random(n_residues)

                    tauE = np.zeros(n_residues)
                    for i in range(n_residues):
                        max_tauE_for_residue = 1.0e-9 * (1.0 - 0.7 * S2s[i])
                        tauE[i] = rng.uniform(0, max_tauE_for_residue)
                    tauE = np.minimum(tauE, 0.4 * tauC)

                    # Rex is defined relative to field 1 (600 MHz)
                    Rex_1 = np.zeros(n_residues)
                    if rng.random() <= 0.5:
                        num_exchange_residues = int(n_residues * rng.uniform(0.10, 0.25))
                        if num_exchange_residues > 0:
                            exchange_indices = rng.choice(n_residues, num_exchange_residues, replace=False)
                            Rex_1[exchange_indices] = rng.uniform(1.0, 25.0, num_exchange_residues)
                    
                    # Rex at field 2 (800 MHz) scales with B0^2
                    Rex_2 = Rex_1 * rex_scaling_factor

                    # Calculate relaxation data for Field 1 (600 MHz)
                    R1_f1 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)
                    R2_f1 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex_1, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)
                    NOE_f1 = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_1, OmeN_1, d_const_sq_1, c_const_sq_1)

                    # Calculate relaxation data for Field 2 (800 MHz)
                    R1_f2 = longitudinal_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)
                    R2_f2 = transverse_relaxation_rate_total(orig_sp_den, tauC, S2s, S2f, tauE, Rex_2, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)
                    NOE_f2 = nuclear_overhauser_effect(orig_sp_den, tauC, S2s, S2f, tauE, OmeH_2, OmeN_2, d_const_sq_2, c_const_sq_2)

                    # Add noise
                    R1_f1_noisy = add_noise(R1_f1, scale=noise_level)
                    R2_f1_noisy = add_noise(R2_f1, scale=noise_level)
                    NOE_f1_noisy = add_noise(NOE_f1, scale=noise_level)
                    
                    R1_f2_noisy = add_noise(R1_f2, scale=noise_level)
                    R2_f2_noisy = add_noise(R2_f2, scale=noise_level)
                    NOE_f2_noisy = add_noise(NOE_f2, scale=noise_level)
                    
                    # Stack features to 6 channels: [R1_600, R2_600, NOE_600, R1_800, R2_800, NOE_800]
                    features_tensor = np.stack([
                        R1_f1_noisy, R2_f1_noisy, NOE_f1_noisy, 
                        R1_f2_noisy, R2_f2_noisy, NOE_f2_noisy
                    ], axis=-1).astype(np.float32)
                    
                    # Label Rex is stored relative to 600 MHz
                    local_labels_tensor = np.stack([tauE * 1e9, S2s, Rex_1], axis=-1).astype(np.float32)
                    global_label_val = tauC * 1e9
                    
                    serialized_example = serialize_example(features_tensor, local_labels_tensor, global_label_val)
                    writer.write(serialized_example)

    print("\n2-Field stratified data generation complete!")

if __name__ == "__main__":
    main()

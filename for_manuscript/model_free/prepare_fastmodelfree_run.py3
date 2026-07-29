#!/usr/bin/env python3
import os, sys
import argparse
import numpy as np 
from tqdm import tqdm

def prepare_fastmodelfree_files(b0, benchmark_file, output_dir_base):
    """
    Splits a large benchmark relaxation data file into a directory structure
    suitable for running FastModelFree, generating a custom config file for each protein.

    Args:
        benchmark_file (str): Path to the input file containing R1, R2, NOE data
                              for multiple proteins, separated by "END".
        output_dir_base (str): Name of the base directory to create output folders in.
    """
    print(f"--- Preparing FastModelFree Run (with dynamic config) ---")
    print(f"Reading benchmark data from: {benchmark_file}")
    # Config template is now used conceptually, not directly copied

    # --- Create the main output directory ---
    os.makedirs(output_dir_base, exist_ok=True)
    print(f"Output directories will be created in: {output_dir_base}")

    # --- Read the benchmark file and process protein by protein ---
    protein_count = 0
    try:
        with open(benchmark_file, "r") as f_in:
            content_peek = f_in.read()
            total_proteins = content_peek.count("END")
            f_in.seek(0)

            protein_data_buffer = []
            pbar = tqdm(total=total_proteins, desc="Processing Proteins")
            for line in f_in:
                if line.strip().upper() == "END":
                    if not protein_data_buffer:
                        continue

                    # --- Process a complete protein ---
                    protein_count += 1
                    protein_dir = os.path.join(output_dir_base, f"protein_{protein_count:05d}")
                    os.makedirs(protein_dir, exist_ok=True)

                    r1_lines = []
                    r2_lines = []
                    noe_lines = []
                    n_residues = len(protein_data_buffer) # Estimate residues from lines

                    try:
                        res_nums = []
                        for data_line in protein_data_buffer:
                            parts = data_line.strip().split()
                            if len(parts) < 7: continue

                            res_num = parts[0]
                            res_nums.append(int(res_num)) # Keep track for accurate count
                            r1, r1e = parts[1], parts[2]
                            r2, r2e = parts[3], parts[4]
                            noe, noee = parts[5], parts[6]

                            r1_lines.append(f"{res_num}\t{r1}\t{r1e}\n")
                            r2_lines.append(f"{res_num}\t{r2}\t{r2e}\n")
                            noe_lines.append(f"{res_num}\t{noe}\t{noee}\n")

                        # Get actual number of residues processed
                        n_residues = len(res_nums)
                        if n_residues == 0:
                            print(f"\nWarning: No valid data found for protein {protein_count}. Skipping.")
                            protein_data_buffer = []
                            pbar.update(1)
                            continue

                        # Write the relaxation data files
                        with open(os.path.join(protein_dir, "R1.txt"), "w") as f_r1: f_r1.writelines(r1_lines)
                        with open(os.path.join(protein_dir, "R2.txt"), "w") as f_r2: f_r2.writelines(r2_lines)
                        with open(os.path.join(protein_dir, "NOE.txt"), "w") as f_noe: f_noe.writelines(noe_lines)

                        # --- Dynamically generate FMF.config content ---
                        tauc_guess = 0.0005998 * n_residues * 110 + 0.1674
                        lower_bound = max(1.0, tauc_guess - 4.0) # Ensure lower bound >= 1.0
                        upper_bound = tauc_guess + 4.0
                        tm_grid_steps = 20 # Keep grid steps reasonable

                        config_content = f"""tensor Isotropic
cutoff 0.95
Fcutoff 0.80
optimize Yes
maxloop 10
almost1 20
S2cutoff 0.0
seed 1985
numsim 300
jobname protein_{protein_count:05d}
gamma -2.710
rNH 1.02
N15CSA -160
tm {tauc_guess:.3f}
tmMin {lower_bound:.3f}
tmMax {upper_bound:.3f}
tmGrid {tm_grid_steps}
tmConv 0.001
Dratio 0.819
DratioMin 0.8
DratioMax 0.9
DratioGrid 5
DratioConv 0.001
Theta 20
ThetaMin 0
ThetaMax 40
ThetaGrid 5
ThetaConv 0.001
Phi 0
PhiMin 0
PhiMax 360
PhiGrid 20
PhiConv 0.001
model1only No
mpdb pdbfile
file{{0}}{{R1}} R1.txt
file{{0}}{{R2}} R2.txt
file{{0}}{{NOE}} NOE.txt
file{{0}}{{field}} {b0:.1f}
"""
                        # Write the dynamic config file
                        with open(os.path.join(protein_dir, "FMF.config"), "w") as f_cfg:
                            f_cfg.write(config_content)

                    except Exception as e:
                        print(f"\nError processing protein {protein_count}: {e}")

                    protein_data_buffer = []
                    pbar.update(1)
                else:
                    protein_data_buffer.append(line)
            pbar.close()

    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{benchmark_file}'")
        return
    except Exception as e:
        print(f"Error reading benchmark file: {e}")
        return

    print(f"\nSuccessfully created directories and files for {protein_count} proteins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a benchmark relaxation data file into a directory structure for FastModelFree.")
    parser.add_argument('-field', type=int, default=800, help='Proton resonance frequency in MHz (e.g., 800)')
    parser.add_argument("-benchmark_file", type=str, help="Path to the modelfree  benchmark file (e.g., 'modelfree_model4_benchmark_data.txt').")
    parser.add_argument("-out_dir", type=str, default="800", help="Name of the base output directory (default: fastmodelfree_runs)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    output_dir_base = os.path.join(args.out_dir,'fmf_model4_runs')
    prepare_fastmodelfree_files(args.field, args.benchmark_file, output_dir_base)



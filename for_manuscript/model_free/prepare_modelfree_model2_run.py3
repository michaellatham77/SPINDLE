#!/usr/bin/env python3
#
# Make directories and input files for modelfree runs
# Only fitting data to model 2
#
# @author: MP Latham
# @date: Jan. 23, 2026
#

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

def prepare_modelfree_files(field, benchmark_file, output_dir_base):
#def prepare_modelfree_files(benchmark_file, output_dir_base="./modelfree_model2_runs"):
    """
    Forces 'mfmodel' to fix Rex=0 (Model 2).
    """
    print("--- Preparing Modelfree Run (Force Model 2) ---")

    os.makedirs(output_dir_base, exist_ok=True)
    print(f"Output directories will be created in: {output_dir_base}")

    protein_count = 0
    with open(benchmark_file, "r") as f_in:
        content = f_in.read()
        total_proteins = content.count("END")
        f_in.seek(0)

        protein_data_buffer = []
        pbar = tqdm(total=total_proteins, desc="Processing Proteins")
        for line in f_in:
            if line.strip() == "END":
                if not protein_data_buffer:
                    continue

                protein_count += 1
                protein_dir = os.path.join(output_dir_base, f"protein_{protein_count:05d}")
                os.makedirs(protein_dir, exist_ok=True)

                n_residues = len(protein_data_buffer)
                mfdata_content = ""
                mfmodel_content = ""
                mfparam_content = ""

                # 1. mfinput (Standard isotropic grid search)
                tauc_guess = 0.0005998 * n_residues * 110 + 0.1674
                lower_bound = max(1.0, tauc_guess - 3.0)
                upper_bound = tauc_guess + 3.0

                mfinput_content = f"""optimization tval
seed 1241656474
search grid
diffusion isotropic grid
algorithm powell grid 1
simulations pred 200 0.00
selection none
sim_algorithm powell grid 1
fields 1 {field:.3f} 
tm       {tauc_guess:.2f} 1  2  {lower_bound:.2f}   {upper_bound:.2f}  20
"""

                # 2. Generate files
                for data_line in protein_data_buffer:
                    parts = data_line.strip().split()
                    res_num, r1, r1e, r2, r2e, noe, noee = parts
                    
                    # mfdata
                    mfdata_content += f"spin  {res_num}\n"
                    mfdata_content += f"R1 {field:.3f}  {r1}  {r1e} 1\n"
                    mfdata_content += f"R2 {field:.3f}  {r2}  {r2e} 1\n"
                    mfdata_content += f"NOE {field:.3f} {noe} {noee} 1\n\n"

                    # --- CRITICAL CHANGE: FORCE MODEL 2 IN MFMODEL ---
                    # 1 = Optimize, 0 = Fix
                    mfmodel_content += f"spin  {res_num}\n"
                    # Local diffusion (unused in isotropic but required syntax)
                    mfmodel_content += "M1 tloc 6.10   0   2      0.000      10.000 20\n"
                    mfmodel_content += "M1 Theta 0.0   0   2      0.000      90.000 20\n"
                    mfmodel_content += "M1 S2f   1.0   0   2      0.000       1.000 20\n"
                    # Optimize S2s (Flag 1)
                    mfmodel_content += "M1 S2s   1.0   1   2      0.000       1.000 20\n"
                    # Optimize te (Flag 1)
                    mfmodel_content += "M1 te    0.0   1   2      0.000    1000.000 20\n"
                    # FIX Rex (Flag 0)
                    mfmodel_content += "M1 Rex   0.0   0   2      0.000      25.000 20\n\n"

                    # mfparam
                    mfparam_content += f"spin  {res_num}\n"
                    mfparam_content += f"constants    {res_num} N15     -2.710     1.020     -160.00\n"
                    mfparam_content += "vector N H\n\n"

                # 3. Write files
                with open(os.path.join(protein_dir, "mfdata"), "w") as f:
                    f.write(mfdata_content)
                with open(os.path.join(protein_dir, "mfmodel"), "w") as f:
                    f.write(mfmodel_content)
                with open(os.path.join(protein_dir, "mfparam"), "w") as f:
                    f.write(mfparam_content)
                with open(os.path.join(protein_dir, "mfinput"), "w") as f:
                    f.write(mfinput_content)

                protein_data_buffer = []
                pbar.update(1)
            else:
                protein_data_buffer.append(line)
        pbar.close()

    print(f"\nSuccessfully created 'Step 1' directories for {protein_count} proteins.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-field", type=int, default=800, help="Proton resonance frequency in MHz (e.g., 800)")
    parser.add_argument("-benchmark_file", type=str, help="Path to modelfree_model2_benchmark_data.txt'")
    parser.add_argument("-out_dir", type=str, default="800", help="Directory for mf4 directories")

    if len(sys.argv) == 1:
            parser.print_help()
            sys.exit()

    args = parser.parse_args()

    output_dir_base = args.out_dir+'/modelfree_model2_runs'
    prepare_modelfree_files(args.field, args.benchmark_file, output_dir_base)

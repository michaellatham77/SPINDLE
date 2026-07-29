#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import sys

# --- Configuration ---
RESULTS_DIR = "modelfree_model2_runs"  
GROUND_TRUTH_FILE = "modelfree_model2_dataset.npz"
OUTPUT_CSV = "modelfree_vs_truth.csv"
MAX_RESIDUES = 300 

def extract_from_star_list(star_file, star_block, star_loop):
    """
    Standard extractor for RESIDUE data (S2, te).
    Finds a block, finds a header loop, and reads the vertical columns.
    """
    output = []
    copy = False
    copy2 = False

    if not os.path.exists(star_file): return []

    try:
        with open(star_file, 'r', encoding='utf-8', errors='ignore') as starfile:
            for line in starfile:
                stripped = line.strip()
                if star_block in line:
                    copy = True
                elif line and line[0].isalpha() and 'loop_' not in line and copy:
                    copy = False
                elif copy:
                    if star_loop in line:
                        copy2 = True
                    elif "stop_" in line:
                        copy2 = False
                    elif copy2:
                        parts = stripped.split()
                        if parts and parts[0].replace('.', '', 1).isdigit():
                            output.append(parts)
    except: return []
    return output

def extract_global_tm(star_file):
    """
    Specific extractor for the Diffusion Tensor block.
    Looks for the row starting with 'tm' and grabs the Fit_Value (Col 2).
    """
    if not os.path.exists(star_file): return np.nan
    
    in_diff_block = False
    
    try:
        with open(star_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                
                # 1. Enter the Diffusion Block
                if "data_diffusion_tensor" in line:
                    in_diff_block = True
                    continue
                
                # 2. Exit Block (if we hit a new data block or stop)
                if in_diff_block and (line.startswith("data_") or line.startswith("stop_")):
                    # Be careful: stop_ ends the loop, but usually we want to stay open until we find tm
                    # But if we see "data_", we are definitely done.
                    if line.startswith("data_"): 
                        in_diff_block = False
                
                # 3. Parse inside the block
                if in_diff_block:
                    parts = stripped.split()
                    if not parts: continue
                    
                    # Look for the row starting with 'tm'
                    if parts[0] == "tm":
                        # Structure: tm (ns) Value Error ...
                        # Index:     0   1     2     3
                        try:
                            val_str = parts[2]
                            return np.nan if '*****' in val_str else float(val_str)
                        except (IndexError, ValueError):
                            return np.nan
    except:
        return np.nan
        
    return np.nan

def check_log_for_errors(log_path):
    """Scans log for specific convergence failures."""
    if not os.path.exists(log_path): return False
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
            if "brent exceed maximum iterations" in content: return True
            if "IEEE_DIVIDE_BY_ZERO" in content: return True
    except: pass
    return False

def parse_protein_folder(folder_path):
    mfout_path = os.path.join(folder_path, "mfout")
    mfmodel_path = os.path.join(folder_path, "mfmodel")
    mfrun_log_path = os.path.join(folder_path, "mfrun.log")
    
    # 1. Check for Successful Output
    if not (os.path.exists(mfout_path) and os.path.getsize(mfout_path) > 0):
        # Failure Analysis
        if os.path.exists(mfmodel_path):
            try:
                with open(mfmodel_path, 'r') as f:
                    lines = [l for l in f.readlines() if l.strip()]
                    n_res = len(lines) // 7
                if n_res > MAX_RESIDUES: return None, None, "Excluded_Size"
            except: pass
            
        if check_log_for_errors(mfrun_log_path): return None, None, "Convergence_Fail"
        if os.path.exists(mfrun_log_path): return None, None, "Convergence_Fail" # Stalled

        return None, None, "Run_Failed"

    # 2. Parse Valid Output
    results = {}
    
    # --- Local Parameters (Ragged List) ---
    s2_list = extract_from_star_list(mfout_path, "data_model_1", "_S2s")
    if not s2_list: s2_list = extract_from_star_list(mfout_path, "data_model_1", " S2s")
    if not s2_list: s2_list = extract_from_star_list(mfout_path, "data_model_1", "_S2")
    
    te_list = extract_from_star_list(mfout_path, "data_model_1", "_te")
    if not te_list: te_list = extract_from_star_list(mfout_path, "data_model_1", " te")

    # --- Global Parameters (Direct Extraction) ---
    fit_tm = extract_global_tm(mfout_path)

    # --- Map Local Data ---
    def map_data(data_list, param_name):
        if not data_list: return
        for row in data_list:
            if len(row) < 2: continue
            try:
                res = int(row[0])
                val_str = row[1]
                val = np.nan if '*****' in val_str else float(val_str)
                if res not in results: results[res] = {}
                results[res][param_name] = val
            except: continue

    map_data(s2_list, 'S2')
    map_data(te_list, 'te')
    
    if not results:
        return None, None, "Parse_Error"
        
    return results, fit_tm, "Success"

# --- Main Execution ---
print(f"Loading Ground Truth: {GROUND_TRUTH_FILE}")
if not os.path.exists(GROUND_TRUTH_FILE):
    print("Error: Ground truth file not found.")
    sys.exit(1)

truth_data = np.load(GROUND_TRUTH_FILE, allow_pickle=True)
truth_labels = truth_data['labels']

print(f"Scanning {RESULTS_DIR}...")
comparison_data = []
counters = {
    "Success": 0, 
    "Run_Failed": 0, 
    "Excluded_Size": 0, 
    "Convergence_Fail": 0,
    "Parse_Error": 0
}

for protein_idx, labels in enumerate(truth_labels):
    p_id = protein_idx + 1
    folder_name = f"protein_{p_id:05d}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    
    fitted_results, fit_tm, status = parse_protein_folder(folder_path)
    
    counters[status] += 1
    
    # Get True Data
    true_s2 = labels['S2']
    true_te = labels['tauE_ns']
    # Handle Global tauC (Scalar or Array)
    true_tauc_val = labels['tauC_ns']
    if isinstance(true_tauc_val, (list, np.ndarray)):
         true_tauc_val = true_tauc_val[0]
    
    for i in range(len(true_s2)):
        res_num = i + 1
        fit_s2_val = np.nan
        fit_te_val = np.nan
        
        if status == "Success" and res_num in fitted_results:
            row = fitted_results[res_num]
            fit_s2_val = row.get('S2', np.nan)
            
            # Unit Conversion: ps -> ns
            raw_te = row.get('te', np.nan)
            if pd.notna(raw_te):
                fit_te_val = raw_te / 1000.0
        
        comparison_data.append({
            "Protein": p_id,
            "Residue": res_num,
            "Status": status,
            "True_S2": true_s2[i],
            "Fit_S2": fit_s2_val,
            "True_tauE": true_te[i],
            "Fit_tauE": fit_te_val,
            "True_tauC": true_tauc_val,
            "Fit_tauC": fit_tm if status == "Success" else np.nan
        })

# --- Final Stats ---
df = pd.DataFrame(comparison_data)

df['S2_Error'] = df['Fit_S2'] - df['True_S2']
df['tauE_Error'] = df['Fit_tauE'] - df['True_tauE']
df['tauC_Error'] = df['Fit_tauC'] - df['True_tauC']

print("\n" + "="*40)
print(f"Processing Complete.")
print(f"Successful Runs:    {counters['Success']}")
print(f"Excluded (>300 res):{counters['Excluded_Size']}")
print(f"Convergence Fail:   {counters['Convergence_Fail']}")
print(f"Other Failures:     {counters['Run_Failed']}")
print("="*40)

if counters['Success'] > 0:
    success_df = df[df['Status'] == 'Success']
    s2_clean = success_df.dropna(subset=['S2_Error'])
    te_clean = success_df.dropna(subset=['tauE_Error'])
    tc_clean = success_df.dropna(subset=['tauC_Error'])
    
    # Drop duplicates for global stats
    unique_proteins = tc_clean.drop_duplicates(subset=['Protein'])

    print("\n--- ERROR METRICS ---")
    if not s2_clean.empty:
        rmse_s2 = np.sqrt((s2_clean['S2_Error']**2).mean())
        print(f"RMSE S2:   {rmse_s2:.5f}")
    if not te_clean.empty:
        rmse_te = np.sqrt((te_clean['tauE_Error']**2).mean())
        print(f"RMSE tauE: {rmse_te:.5f} ns")
    if not unique_proteins.empty:
        rmse_tc = np.sqrt((unique_proteins['tauC_Error']**2).mean())
        print(f"RMSE tauC: {rmse_tc:.5f} ns")

df.to_csv(OUTPUT_CSV, index=False)

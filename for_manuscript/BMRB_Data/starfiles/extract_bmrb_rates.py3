#!/usr/bin/env python3

import os
import argparse
import re
import pandas as pd

# ==========================================
# 1. DATA EXTRACTION FUNCTIONS
# ==========================================
def extract_metadata(filepath):
    entry_id = os.path.basename(filepath).split('.')[0]
    doi = "Not Found"
    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('_Entry.ID'):
                    parts = stripped.split()
                    if len(parts) > 1 and parts[1] != '.': 
                        entry_id = parts[1]
                if stripped.startswith('_Citation.DOI'):
                    parts = stripped.split()
                    if len(parts) > 1 and parts[1] != '.': 
                        doi = parts[1]
    except Exception:
        pass
    return entry_id, doi

def extract_multi_field_dynamics(filepath):
    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()
        
    def snap_field(raw_str):
        try:
            val = float(raw_str)
            supported = [500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 1100.0]
            closest = min(supported, key=lambda x: abs(x - val))
            if abs(closest - val) <= 20.0:
                return str(closest)
            return str(val)
        except ValueError:
            return str(raw_str)
            
    def clean_cond(val):
        """Sanitizes condition labels so they are safe for file names."""
        return re.sub(r'[^A-Za-z0-9_\-]', '_', str(val).strip('\'"'))

    def get_cond(row_dict, prefix, fallback_cond):
        """Extracts the sample condition to prevent overriding data at the same field."""
        if f'_{prefix}.Sample_condition_list_label' in row_dict and row_dict[f'_{prefix}.Sample_condition_list_label'] != '.':
            return clean_cond(row_dict[f'_{prefix}.Sample_condition_list_label'])
        if f'_{prefix}.Sample_condition_list_ID' in row_dict and row_dict[f'_{prefix}.Sample_condition_list_ID'] != '.':
            return "Cond_" + clean_cond(row_dict[f'_{prefix}.Sample_condition_list_ID'])
        if f'_{prefix}.Sample_label' in row_dict and row_dict[f'_{prefix}.Sample_label'] != '.':
            return clean_cond(row_dict[f'_{prefix}.Sample_label'])
        return fallback_cond
            
    relax_data = {}
    current_field = "Unknown"
    current_save_cond = "Cond_1"
    in_loop = False
    in_data = False
    current_loop_tags = []
    current_unit = "s-1" 
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        if stripped.startswith('save_'):
            current_unit = "s-1"
            current_save_cond = "Cond_1"
            
        if stripped.startswith('loop_'):
            current_loop_tags = []
            in_loop = True
            in_data = False
            continue
            
        # Capture metadata outside the loop
        if not in_loop and stripped.startswith('_'):
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1 and parts[1] != '.':
                tag = parts[0]
                val = parts[1].strip('\'"')
                
                if 'Spectrometer_frequency_1H' in tag:
                    current_field = snap_field(val)
                elif '.Val_units' in tag or '.T2_val_units' in tag or '.T1_val_units' in tag:
                    current_unit = val.lower()
                elif 'Sample_condition_list_label' in tag:
                    current_save_cond = clean_cond(val)
                elif 'Sample_condition_list_ID' in tag:
                    current_save_cond = "Cond_" + clean_cond(val)
                elif 'Sample_label' in tag:
                    current_save_cond = clean_cond(val)
            continue
            
        if in_loop and stripped.startswith('_'):
            current_loop_tags.append(stripped)
            continue
            
        if in_loop and current_loop_tags and not stripped.startswith('_') and not stripped.startswith('stop_'):
            in_data = True
            
        if in_data and stripped == 'stop_':
            in_loop = False
            in_data = False
            current_loop_tags = []
            continue
            
        if in_data:
            parts = stripped.split()
            if len(parts) >= len(current_loop_tags):
                row_dict = {tag: val for tag, val in zip(current_loop_tags, parts)}
                
                field_t1 = snap_field(row_dict.get('_T1.Spectrometer_frequency_1H', current_field)) if row_dict.get('_T1.Spectrometer_frequency_1H', '.') != '.' else current_field
                field_t2 = snap_field(row_dict.get('_T2.Spectrometer_frequency_1H', current_field)) if row_dict.get('_T2.Spectrometer_frequency_1H', '.') != '.' else current_field
                field_noe = snap_field(row_dict.get('_Heteronucl_NOE.Spectrometer_frequency_1H', current_field)) if row_dict.get('_Heteronucl_NOE.Spectrometer_frequency_1H', '.') != '.' else current_field
                
                cond_t1 = get_cond(row_dict, "T1", current_save_cond)
                cond_t2 = get_cond(row_dict, "T2", current_save_cond)
                cond_noe = get_cond(row_dict, "Heteronucl_NOE", current_save_cond)
                
                # Extract R1
                if '_T1.Val' in row_dict and '_T1.Seq_ID' in row_dict and row_dict['_T1.Val'] != '.':
                    val = float(row_dict['_T1.Val'])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        seq_id = int(row_dict['_T1.Seq_ID'])
                        relax_data.setdefault((field_t1, cond_t1), {}).setdefault(seq_id, {})['R1'] = val
                    
                # Extract R2
                elif '_T2.T2_val' in row_dict and '_T2.Seq_ID' in row_dict and row_dict['_T2.T2_val'] != '.':
                    val = float(row_dict['_T2.T2_val'])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        seq_id = int(row_dict['_T2.Seq_ID'])
                        relax_data.setdefault((field_t2, cond_t2), {}).setdefault(seq_id, {})['R2'] = val
                    
                # Extract NOE
                elif '_Heteronucl_NOE.Val' in row_dict and '_Heteronucl_NOE.Seq_ID_1' in row_dict and row_dict['_Heteronucl_NOE.Val'] != '.':
                    val = float(row_dict['_Heteronucl_NOE.Val'])
                    seq_id = int(row_dict['_Heteronucl_NOE.Seq_ID_1'])
                    relax_data.setdefault((field_noe, cond_noe), {}).setdefault(seq_id, {})['NOE'] = val

    dfs_relax = {}
    for (field, cond), data_dict in relax_data.items():
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        if not df.empty:
            df.index.name = 'residue_number'
            df.sort_index(inplace=True)
            cols = [c for c in ['R1', 'R2', 'NOE'] if c in df.columns]
            dfs_relax[(field, cond)] = df[cols]
            
    # --- FOOLPROOF RELAXATION FALLBACK HEURISTIC ---
    for (field, cond), df in dfs_relax.items():
        median_r1 = df['R1'].median() if 'R1' in df.columns else None
        median_r2 = df['R2'].median() if 'R2' in df.columns else None
        
        if median_r2 is not None and median_r2 < 1.0:
            print(f"Auto-Correction [{field} MHz | {cond}]: Median R2 is {median_r2:.3f}. Converting times to rates...")
            df['R2'] = 1.0 / df['R2']
            if 'R1' in df.columns: df['R1'] = 1.0 / df['R1']
            
        elif median_r1 is not None and median_r1 > 50.0:
            print(f"Auto-Correction [{field} MHz | {cond}]: Median R1 is {median_r1:.0f}. Converting ms times to s-1 rates...")
            if 'R1' in df.columns: df['R1'] = 1000.0 / df['R1']
            if 'R2' in df.columns: df['R2'] = 1000.0 / df['R2']

    return dfs_relax

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def process_single_file(filepath, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    entry_id, doi = extract_metadata(filepath)
    print(f"\nProcessing Entry: {entry_id} (DOI: {doi})")
    
    dfs_relax = extract_multi_field_dynamics(filepath)
    
    if not dfs_relax:
        print(" No relaxation data found in this file.")
        return

    extracted_files = 0
    for (field, cond), df in dfs_relax.items():
        # Check if we have all three required measurements
        missing_cols = [c for c in ['R1', 'R2', 'NOE'] if c not in df.columns]
        if missing_cols:
            print(f"Skipping {field} MHz ({cond}): Missing {', '.join(missing_cols)}")
            continue
            
        # Drop residues that don't have all three values
        df_clean = df.dropna(subset=['R1', 'R2', 'NOE']).copy()
        
        if df_clean.empty:
            print(f"Skipping {field} MHz ({cond}): No residues with overlapping R1, R2, and NOE.")
            continue
            
        # Format explicitly for SPINDLE standalone
        df_clean = df_clean[['R1', 'R2', 'NOE']]
        
        out_name = f"{entry_id}_{field}MHz_{cond}.txt"
        out_path = os.path.join(out_dir, out_name)
        
        # Save as space-delimited text file
        df_clean.to_csv(out_path, sep=' ', header=True, index=True)
        print(f"Saved {len(df_clean)} residues to: {out_path}")
        extracted_files += 1
        
    if extracted_files == 0:
        print("\nNo complete data sets (R1, R2, NOE) were extracted.")
    else:
        print(f"\nSuccessfully extracted {extracted_files} formatted input file(s) to '{out_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BMRB STAR files into space-delimited text files for standalone SPINDLE fitting.")
    parser.add_argument("-in", "--input", required=True, help="Path to the individual BMRB NMR-STAR file (.str or .txt).")
    parser.add_argument("-out", "--out_dir", default="bmrb_extracted_data", help="Directory to save the formatted text files (default: bmrb_extracted_data)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Could not find file '{args.input}'")
    else:
        process_single_file(args.input, args.out_dir)

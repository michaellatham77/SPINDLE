#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.stats import pearsonr

def clean_tauc(val):
    """Extracts the first valid floating point or integer number from a messy string."""
    if pd.isna(val): return np.nan
    val_str = str(val).replace('\n', '')
    match_dec = re.search(r'(\d+\.\d+)', val_str)
    if match_dec: return float(match_dec.group(1))
    match_int = re.search(r'(\d+)', val_str)
    if match_int: return float(match_int.group(1))
    return np.nan

def extract_bmrb_id(val):
    """Extracts just the numeric BMRB ID to ensure clean merging."""
    match = re.search(r'(\d+)', str(val))
    return match.group(1) if match else str(val)

def extract_field_cond(field_cond_str):
    """Splits SPINDLE's '600MHz_Cond_1' into (600.0, 'Cond_1')."""
    if "Averaged" in str(field_cond_str):
        return np.nan, "Averaged"
    match = re.match(r'^([\d\.]+)MHz_(.+)$', str(field_cond_str))
    if match:
        return float(match.group(1)), match.group(2)
    return np.nan, str(field_cond_str)

def conditions_match(exp_cond, sp_cond):
    """Text-based matching logic between the Excel column and SPINDLE's extracted label."""
    exp_c = str(exp_cond).strip().lower()
    sp_c = str(sp_cond).strip().lower()
    
    if exp_c == sp_c: return True
    if exp_c in ["", "nan"] and sp_c == "cond_1": return True
    
    # Keyword / Substring matching (e.g., 'condition 4' vs 'cond_4')
    if 'condition' in exp_c and 'cond_' in sp_c:
        num_exp = re.search(r'\d+', exp_c)
        num_sp = re.search(r'\d+', sp_c)
        if num_exp and num_sp and num_exp.group() == num_sp.group():
            return True

    # Broad substring overlap (e.g., 'apo' in 'apo_bd2')
    if exp_c not in ["", "nan"] and (exp_c in sp_c or sp_c in exp_c): return True
    
    # Safe sanitized names check
    safe_exp = re.sub(r'[^a-z0-9_\-]', '_', exp_c)
    if safe_exp not in ["", "_", "nan"] and (safe_exp in sp_c or sp_c in safe_exp): 
        return True
        
    return False

def calculate_metrics(y_true, y_pred):
    """Calculates statistics between two series."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) < 2: return float('nan'), float('nan'), float('nan'), 0
        
    r_val, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r_val, rmse, mae, len(y_true)

def main(spindle_csv, exp_csv, out_dir, target_sheet):
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(" 🛠️  DIAGNOSTIC DATA PIPELINE INITIALIZING")
    print("="*50)
    
    # --- LOAD SPINDLE ---
    try: 
        df_spindle = pd.read_csv(spindle_csv)
        print(f"✅ Loaded SPINDLE CSV: {len(df_spindle)} rows.")
    except Exception as e:
        print(f"❌ Error reading SPINDLE CSV: {e}"); return

    # --- LOAD EXCEL/CSV ---
    try: 
        if exp_csv.lower().endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(exp_csv)
            # Fallback logic to accept any available sheet
            if target_sheet not in xls.sheet_names:
                fallback_sheet = xls.sheet_names[0]
                print(f"⚠️ Sheet '{target_sheet}' not found. Automatically falling back to first available: '{fallback_sheet}'")
                target_sheet = fallback_sheet
                
            df_exp = pd.read_excel(exp_csv, sheet_name=target_sheet)
            print(f"✅ Loaded Excel File (Using sheet: '{target_sheet}'): {len(df_exp)} rows.")
        else:
            df_exp = pd.read_csv(exp_csv)
            print(f"✅ Loaded Experimental CSV: {len(df_exp)} rows.")
    except Exception as e:
        print(f"❌ Error reading Experimental file: {e}\n(Ensure 'openpyxl' is installed!)")
        return

    # --- PARSE SPINDLE DATA ---
    df_spindle['BMRB_ID'] = df_spindle['Entry_ID'].apply(extract_bmrb_id)
    df_spindle[['Field_MHz', 'Spindle_Cond']] = df_spindle.apply(
        lambda row: pd.Series(extract_field_cond(row['Field_Condition'])), axis=1
    )
    df_spindle_fields = df_spindle[(df_spindle['Spindle_Cond'] != 'Averaged')].dropna(subset=['TauC_Pred_ns'])
    spindle_ids = set(df_spindle_fields['BMRB_ID'].unique())
    print(f"   -> SPINDLE contains {len(spindle_ids)} unique valid proteins.")

    # --- PARSE EXPERIMENTAL DATA ---
    if 'NMR Star Number' not in df_exp.columns or 'TauC' not in df_exp.columns:
        print("❌ Error: Missing required columns 'NMR Star Number' or 'TauC' in Experimental file.")
        return
        
    df_exp['BMRB_ID'] = df_exp['NMR Star Number'].apply(extract_bmrb_id)
    df_exp['TauC_Exp_ns'] = df_exp['TauC'].apply(clean_tauc)
    df_exp = df_exp.dropna(subset=['TauC_Exp_ns'])
    
    cond_col_name = df_exp.columns[6] if len(df_exp.columns) >= 7 else None
    df_exp['Excel_Condition'] = df_exp[cond_col_name].fillna('').astype(str).str.strip() if cond_col_name else ""
    
    df_exp['Is_Isotropic'] = df_exp.astype(str).apply(lambda x: x.str.contains('isotropic', case=False, na=False)).any(axis=1)
    
    exp_ids = set(df_exp['BMRB_ID'].unique())
    print(f"   -> Experimental File contains {len(exp_ids)} unique valid proteins.")

    # --- INTERSECTION CHECK ---
    overlapping_ids = spindle_ids.intersection(exp_ids)
    print(f"   -> Found {len(overlapping_ids)} overlapping proteins between SPINDLE and Excel.")
    
    if len(overlapping_ids) == 0:
        print("\n❌ FATAL: No intersecting BMRB IDs found. Nothing to compare.")
        return

    print("\n🔄 Commencing Topology Mapping & Text Matching...")
    matched_records = []
    
    for b_id in overlapping_ids:
        sp_matches = df_spindle_fields[df_spindle_fields['BMRB_ID'] == b_id]
        ex_matches = df_exp[df_exp['BMRB_ID'] == b_id]
        
        unique_sp_conds = sp_matches['Spindle_Cond'].unique()
        unique_ex_conds = ex_matches['Excel_Condition'].unique()
        
        # TOPOLOGY MATCH: If SPINDLE found 1 condition, and Excel lists 1 condition -> Pair them automatically
        if len(unique_sp_conds) == 1 and len(unique_ex_conds) == 1:
            e_row = ex_matches.iloc[0]
            exp_cond_text = e_row['Excel_Condition']
            for _, sp_row in sp_matches.iterrows():
                matched_records.append({
                    'BMRB_ID': b_id,
                    'Protein': e_row.get('Protein', 'Unknown'),
                    'Excel_Condition_Provided': exp_cond_text if exp_cond_text else "Default/Forced",
                    'TauC_Exp_ns': e_row['TauC_Exp_ns'],
                    'Spindle_Field_MHz': sp_row['Field_MHz'],
                    'Spindle_Condition_Found': sp_row['Spindle_Cond'],
                    'TauC_Pred_ns': sp_row['TauC_Pred_ns'],
                    'S2_Pearson_r': sp_row.get('S2_R', np.nan),
                    'Is_Isotropic': e_row['Is_Isotropic'],
                    'Match_Type': 'Topology (1:1)'
                })
            continue
            
        # STRICT TEXT MATCH: Multiple conditions exist, so use regex/text matching
        for _, e_row in ex_matches.iterrows():
            exp_cond = e_row['Excel_Condition']
            for _, sp_row in sp_matches.iterrows():
                if conditions_match(exp_cond, sp_row['Spindle_Cond']):
                    matched_records.append({
                        'BMRB_ID': b_id,
                        'Protein': e_row.get('Protein', 'Unknown'),
                        'Excel_Condition_Provided': exp_cond,
                        'TauC_Exp_ns': e_row['TauC_Exp_ns'],
                        'Spindle_Field_MHz': sp_row['Field_MHz'],
                        'Spindle_Condition_Found': sp_row['Spindle_Cond'],
                        'TauC_Pred_ns': sp_row['TauC_Pred_ns'],
                        'S2_Pearson_r': sp_row.get('S2_R', np.nan),
                        'Is_Isotropic': e_row['Is_Isotropic'],
                        'Match_Type': 'Text Match'
                    })

    df_merged = pd.DataFrame(matched_records)
    
    if df_merged.empty:
        print("⚠️ Intersecting BMRB IDs exist, but no conditions could be matched. Check naming conventions.")
        return

    df_merged = df_merged.dropna(subset=['TauC_Pred_ns', 'TauC_Exp_ns'])

    # --- SAVE OUTPUTS ---
    out_csv = os.path.join(out_dir, "TauC_Strict_Matched_Data.csv")
    df_merged.to_csv(out_csv, index=False)
    
    r_val, rmse, mae, n_pts = calculate_metrics(df_merged['TauC_Exp_ns'], df_merged['TauC_Pred_ns'])
    stats_text = f"Total Data Points = {n_pts}\nPearson $r$ = {r_val:.3f}\nRMSE = {rmse:.2f} ns\nMAE = {mae:.2f} ns"
    
    print("\n" + "="*50)
    print(f"🎉 SUCCESS! {n_pts} Conditions Successfully Paired.")
    print("="*50)
    print(f"   Pearson r : {r_val:.3f}")
    print(f"   RMSE      : {rmse:.3f} ns")
    print(f"   MAE       : {mae:.3f} ns\n")

    out_txt = os.path.join(out_dir, "TauC_Global_Statistics.txt")
    with open(out_txt, 'w') as f:
        f.write("=== GLOBAL SPINDLE TAUC STATISTICS ===\n")
        f.write(f"Total Valid Fields/Conditions Mapped: {n_pts}\n")
        f.write(f"Pearson r Correlation Coefficient: {r_val:.3f}\n")
        f.write(f"Root Mean Square Error (RMSE): {rmse:.3f} ns\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.3f} ns\n")

    # --- PLOTTING ---
    plt.figure(figsize=(8, 8))
    
    df_iso = df_merged[df_merged['Is_Isotropic'] == True]
    df_aniso = df_merged[df_merged['Is_Isotropic'] == False]
    
    if not df_iso.empty:
        plt.scatter(df_iso['TauC_Exp_ns'], df_iso['TauC_Pred_ns'], 
                    alpha=0.7, color='dodgerblue', marker='o', edgecolors='k', s=80, label='Isotropic')
    
    # MODIFIED: Changed marker from '^' to 'o' so shapes are uniform, but colors remain different
    if not df_aniso.empty:
        plt.scatter(df_aniso['TauC_Exp_ns'], df_aniso['TauC_Pred_ns'], 
                    alpha=0.7, color='crimson', marker='o', edgecolors='k', s=80, label='Anisotropic / Other')

    min_val = min(df_merged['TauC_Exp_ns'].min(), df_merged['TauC_Pred_ns'].min()) * 0.9
    max_val = max(df_merged['TauC_Exp_ns'].max(), df_merged['TauC_Pred_ns'].max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Parity (y=x)')
    
    plt.title(f"Global Rotational Correlation Time ($\\tau_c$)\nExperimental vs. SPINDLE Predicted (Sheet: {target_sheet})", fontsize=14, fontweight='bold')
    plt.xlabel("Experimental $\\tau_c$ (ns)", fontsize=12)
    plt.ylabel("SPINDLE Predicted $\\tau_c$ (ns)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='gray')
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    out_pdf = os.path.join(out_dir, "TauC_Condition_Comparison_Plot.pdf")
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"📂 All results, CSVs, and Plots saved to: ./{out_dir}/\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map SPINDLE Field/Conditions to Excel TauC experimental data.")
    parser.add_argument("-spindle", required=True, help="Path to the SPINDLE Summary_Stats.csv")
    parser.add_argument("-exp", required=True, help="Path to the Experimental CSV or XLSX file")
    parser.add_argument("-out", default="tauc_comparisons", help="Directory to save the outputs")
    parser.add_argument("-sheet", default="Curated", help="Name of the specific Excel sheet to load (Default: 'Curated')")
    
    args = parser.parse_args()
    main(args.spindle, args.exp, args.out, args.sheet)

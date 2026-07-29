#!/usr/bin/env python3

import os
import sys
import argparse
import re
import zipfile
import tarfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from spindle_calibrations import CALIBRATIONS, MULTIPLIERS

# Force CPU for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
                    if len(parts) > 1 and parts[1] != '.': entry_id = parts[1]
                if stripped.startswith('_Citation.DOI'):
                    parts = stripped.split()
                    if len(parts) > 1 and parts[1] != '.': doi = parts[1]
    except Exception: pass
    return entry_id, doi

def get_seq_id(row_dict, prefix, suffix=""):
    for tag in [f'_{prefix}.Seq_ID{suffix}', f'_{prefix}.Comp_index_ID{suffix}', f'_{prefix}.Auth_seq_ID{suffix}']:
        if tag in row_dict and row_dict[tag] != '.':
            try: return int(row_dict[tag])
            except ValueError: pass
    return None

def extract_multi_field_dynamics(filepath):
    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()
        
    def snap_field(raw_str):
        try:
            val = float(raw_str)
            supported = [500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 1100.0]
            closest = min(supported, key=lambda x: abs(x - val))
            if abs(closest - val) <= 20.0: return str(closest)
            return str(val)
        except ValueError: return str(raw_str)
            
    def clean_cond(val):
        return re.sub(r'[^A-Za-z0-9_\-]', '_', str(val).strip('\'"'))

    def get_cond(row_dict, prefix, fallback_cond):
        if f'_{prefix}.Sample_condition_list_label' in row_dict and row_dict[f'_{prefix}.Sample_condition_list_label'] != '.':
            return clean_cond(row_dict[f'_{prefix}.Sample_condition_list_label'])
        if f'_{prefix}.Sample_condition_list_ID' in row_dict and row_dict[f'_{prefix}.Sample_condition_list_ID'] != '.':
            return "Cond_" + clean_cond(row_dict[f'_{prefix}.Sample_condition_list_ID'])
        if f'_{prefix}.Sample_label' in row_dict and row_dict[f'_{prefix}.Sample_label'] != '.':
            return clean_cond(row_dict[f'_{prefix}.Sample_label'])
        return fallback_cond
            
    relax_data, s2_data = {}, {}
    current_field, current_save_cond, current_unit, file_taue_unit = "Unknown", "Cond_1", "s-1", None
    in_loop, in_data, current_loop_tags = False, False, []
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'): continue
            
        if stripped.startswith('save_'):
            current_unit, current_save_cond = "s-1", "Cond_1"
            
        if stripped.startswith('loop_'):
            current_loop_tags, in_loop, in_data = [], True, False
            continue
            
        if not in_loop and stripped.startswith('_'):
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1 and parts[1] != '.':
                tag, val = parts[0], parts[1].strip('\'"')
                if 'Spectrometer_frequency_1H' in tag: current_field = snap_field(val)
                elif '.Val_units' in tag or '.T2_val_units' in tag or '.T1_val_units' in tag: current_unit = val.lower()
                elif '.Tau_e_val_units' in tag: file_taue_unit = val.lower()
                elif 'Sample_condition_list_label' in tag: current_save_cond = clean_cond(val)
                elif 'Sample_condition_list_ID' in tag: current_save_cond = "Cond_" + clean_cond(val)
                elif 'Sample_label' in tag: current_save_cond = clean_cond(val)
            continue
            
        if in_loop and stripped.startswith('_'):
            current_loop_tags.append(stripped)
            continue
            
        if in_loop and current_loop_tags and not stripped.startswith('_') and not stripped.startswith('stop_'):
            in_data = True
            
        if in_data and stripped == 'stop_':
            in_loop, in_data, current_loop_tags = False, False, []
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
                cond_s2 = get_cond(row_dict, "Order_param", current_save_cond)
                
                # Extract T1
                seq_id_t1 = get_seq_id(row_dict, "T1")
                if '_T1.Val' in row_dict and row_dict['_T1.Val'] != '.' and seq_id_t1 is not None:
                    val = float(row_dict['_T1.Val'])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        relax_data.setdefault((field_t1, cond_t1), {}).setdefault(seq_id_t1, {})['R1'] = val
                    
                # Extract T2
                seq_id_t2 = get_seq_id(row_dict, "T2")
                t2_val_tag = '_T2.T2_val' if '_T2.T2_val' in row_dict else '_T2.Val'
                if t2_val_tag in row_dict and row_dict[t2_val_tag] != '.' and seq_id_t2 is not None:
                    val = float(row_dict[t2_val_tag])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        relax_data.setdefault((field_t2, cond_t2), {}).setdefault(seq_id_t2, {})['R2'] = val
                    
                # Extract NOE
                seq_id_noe = get_seq_id(row_dict, "Heteronucl_NOE", suffix="_1") or get_seq_id(row_dict, "Heteronucl_NOE")
                if '_Heteronucl_NOE.Val' in row_dict and row_dict['_Heteronucl_NOE.Val'] != '.' and seq_id_noe is not None:
                    relax_data.setdefault((field_noe, cond_noe), {}).setdefault(seq_id_noe, {})['NOE'] = float(row_dict['_Heteronucl_NOE.Val'])
                
                # Extract Model-Free Grouped by Condition
                seq_id_s2 = get_seq_id(row_dict, "Order_param")
                if seq_id_s2 is not None:
                    if '_Order_param.Order_param_val' in row_dict and row_dict['_Order_param.Order_param_val'] != '.':
                        s2_data.setdefault(cond_s2, {}).setdefault(seq_id_s2, {})['S2'] = float(row_dict['_Order_param.Order_param_val'])
                    if '_Order_param.Tau_e_val' in row_dict and row_dict['_Order_param.Tau_e_val'] != '.':
                        s2_data.setdefault(cond_s2, {}).setdefault(seq_id_s2, {})['Tau_e'] = float(row_dict['_Order_param.Tau_e_val'])
                    if '_Order_param.Rex_val' in row_dict and row_dict['_Order_param.Rex_val'] != '.':
                        s2_data.setdefault(cond_s2, {}).setdefault(seq_id_s2, {})['Rex'] = float(row_dict['_Order_param.Rex_val'])

    # Compile Relax Data
    dfs_relax = {}
    for (field, cond), data_dict in relax_data.items():
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        if not df.empty:
            df.index.name = 'Residue'
            df.sort_index(inplace=True)
            cols = [c for c in ['R1', 'R2', 'NOE'] if c in df.columns]
            dfs_relax[(field, cond)] = df[cols]
            
    # Fix Units
    for (field, cond), df in dfs_relax.items():
        median_r1 = df['R1'].median() if 'R1' in df.columns else None
        median_r2 = df['R2'].median() if 'R2' in df.columns else None
        if median_r2 is not None and median_r2 < 1.0:
            df['R2'] = 1.0 / df['R2']
            if 'R1' in df.columns: df['R1'] = 1.0 / df['R1']
        elif median_r1 is not None and median_r1 > 50.0:
            if 'R1' in df.columns: df['R1'] = 1000.0 / df['R1']
            if 'R2' in df.columns: df['R2'] = 1000.0 / df['R2']

    # Compile Model-Free Data per Condition
    dfs_s2 = {}
    for cond, data_dict in s2_data.items():
        df_s2 = pd.DataFrame.from_dict(data_dict, orient='index')
        if not df_s2.empty:
            df_s2.index.name = 'Residue_Number'
            df_s2.sort_index(inplace=True)
            if 'Tau_e' in df_s2.columns:
                df_s2['Tau_e'] = df_s2['Tau_e'].replace(0.0, np.nan)
                valid_taue = df_s2['Tau_e'].dropna()
                if not valid_taue.empty:
                    median_taue = valid_taue.median()
                    if median_taue < 1e-6: df_s2['Tau_e'] = df_s2['Tau_e'] * 1e12
                    elif median_taue < 5.0 and file_taue_unit not in ['ps', 'psec', 'picoseconds']: df_s2['Tau_e'] = df_s2['Tau_e'] * 1000.0
                    elif file_taue_unit in ['ns', 'nsec', 'nanoseconds'] and median_taue < 100.0: df_s2['Tau_e'] = df_s2['Tau_e'] * 1000.0
            dfs_s2[cond] = df_s2

    return dfs_relax, dfs_s2

# ==========================================
# 2. SPINDLE INFERENCE FUNCTIONS
# ==========================================
def load_ensemble(field_mhz, base_model_dir):
    model_dir = os.path.join(base_model_dir, str(int(float(field_mhz))))
    if not os.path.exists(model_dir): return None
    models = []
    for m_file in sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')]):
        try: models.append(tf.keras.models.load_model(os.path.join(model_dir, m_file), compile=False))
        except Exception: pass
    return models

def apply_calibration(pred_mean, pred_err, calib, mult):
    """Applies the polynomial corrections derived from synthetic testing"""
    # S2
    s2_corr = (pred_mean[:, 1] * calib['S2']['slope']) + calib['S2']['intercept']
    s2_corr = np.clip(s2_corr, 0.0, 1.0)
    s2_corr_err = pred_err[:,1] / calib['S2']['slope'] * mult['S2']['mult']

    # tauE
    tauE_ps = pred_mean[:, 0] * 1000.0
    tauE_corr_ps = (tauE_ps * calib['TAUE']['slope']) + calib['TAUE']['intercept']
    tauE_corr_ps = np.maximum(tauE_corr_ps, 0)
    tauE_corr_err = pred_err[:,0] * 1000.0 / calib['TAUE']['slope'] * mult['TAUE']['mult']

    # Rex
    rex = pred_mean[:, 2]
    rex_corr = (calib['REX']['a'] * rex**2) + (calib['REX']['b'] * rex) + calib['REX']['c']
    rex_corr = np.maximum(rex_corr, 0.0)

    rex_derivative = np.abs(2 * calib['REX']['a'] * rex + calib['REX']['b'])
    rex_derivative = np.where(rex_derivative == 0, 1e-6, rex_derivative) # Avoid divide by zero
    rex_corr_err = pred_err[:,2] / rex_derivative * mult['REX']['mult']

    # FIXED: Added correct error variable for rex to return stack
    return np.stack([tauE_corr_ps, s2_corr, rex_corr], axis=1), np.stack([tauE_corr_err, s2_corr_err, rex_corr_err], axis=1)

def run_spindle(df_relax, field, models):
    required_cols = ['R1', 'R2', 'NOE']
    missing_cols = [c for c in required_cols if c not in df_relax.columns]
    if missing_cols: return None
        
    df = df_relax.dropna(subset=required_cols).copy()
    if df.empty: return None
    
    residues = df.index.values
    data = df[['R1', 'R2', 'NOE']].values
    input_tensor = np.expand_dims(data, axis=0).astype(np.float32)
    
    tauc_preds, local_preds = [], []
    for model in models:
        l_pred, g_pred = model.predict(input_tensor, verbose=0)
        tauc_preds.append(g_pred[0, 0])
        local_preds.append(l_pred[0])
        
    calib = CALIBRATIONS[float(field)]
    mult = MULTIPLIERS[float(field)]

    tauc_mean = (np.mean(tauc_preds) - calib['TAUC']['intercept']) / calib['TAUC']['slope']
    tauc_std = np.std(tauc_preds) / calib['TAUC']['slope'] * mult['TAUC']['mult'] # Note: 'slope' key used here might need review based on your MULTIPLIERS dict

    local_mean, local_err = np.mean(local_preds, axis=0), np.std(local_preds, axis=0)
    
    # FIXED: error_mult variable was undefined, replaced with mult
    local_corrected_mean, local_corrected_err = apply_calibration(local_mean, local_err, calib, mult)
    
    results = []
    for i, res in enumerate(residues):
        # FIXED: Mapped mean arrays and generated error outputs
        results.append({
            'Residue': res,
            'S2_pred': local_corrected_mean[i, 1],
            'S2_err': local_corrected_err[i, 1],
            'Tau_e_pred': local_corrected_mean[i, 0],
            'Tau_e_err': local_corrected_err[i, 0],
            'Rex_pred': local_corrected_mean[i, 2],
            'Rex_err': local_corrected_err[i, 2],
            'tauC_global': tauc_mean
        })
    return pd.DataFrame(results)

# ==========================================
# 3. COMBINED PLOTTING FUNCTION
# ==========================================
def plot_combined_profile(df_exp, spindle_dfs, title_prefix, out_dir, metric='S2'):
    plt.figure(figsize=(14, 6))
    
    exp_col = metric
    pred_col = f"{metric}_pred"
    err_col = f"{metric}_err"  # Dynamic reference to error column
    
    # 1. Experimental BMRB Line
    has_exp = False
    if df_exp is not None and not df_exp.empty and exp_col in df_exp.columns:
        df_exp_clean = df_exp.dropna(subset=[exp_col]).copy()
        if not df_exp_clean.empty:
            has_exp = True
            # Reindex to insert NaNs where residues are missing, breaking the line!
            min_r = int(df_exp_clean.index.min())
            max_r = int(df_exp_clean.index.max())
            full_index = pd.Index(range(min_r, max_r + 1), name='Residue_Number')
            df_exp_gapped = df_exp_clean.reindex(full_index)
            
            plt.plot(df_exp_gapped.index, df_exp_gapped[exp_col], 
                     color='black', linestyle='-', linewidth=2, 
                     label=f'Experimental ${metric}$ (BMRB)', zorder=4)

    # 2. SPINDLE Prediction Dots (With Error Bars)
    colors = plt.cm.tab10.colors  
    color_idx = 0
    all_residues = []
    
    for df_pred in spindle_dfs:
        field = df_pred['Field'].iloc[0]
        cond = df_pred['Condition'].iloc[0]
        mean_tauc = df_pred['tauC_global'].mean()
        
        df_plot = df_pred.sort_values(by='Residue').copy()
        
        # Explicitly treat exactly 0.0 as a broken/missing prediction
        df_plot.loc[df_plot[pred_col] == 0.0, pred_col] = np.nan
        
        # Create full residue sequence to force NaN gaps
        if not df_plot.empty:
            min_res = int(df_plot['Residue'].min())
            max_res = int(df_plot['Residue'].max())
            full_index_df = pd.DataFrame({'Residue': range(min_res, max_res + 1)})
            df_plot = pd.merge(full_index_df, df_plot, on='Residue', how='left')
        
        all_residues.extend(df_plot['Residue'].dropna().tolist())
        label_str = f"SPINDLE ({field} MHz | {cond}) - $\\tau_c$: {mean_tauc:.1f} ns"
        
        # FIXED: Use plt.errorbar to properly render caps and stems 
        y_err = df_plot[err_col] if err_col in df_plot.columns else None
        
        plt.errorbar(df_plot['Residue'], df_plot[pred_col], yerr=y_err,
                     color=colors[color_idx % len(colors)], marker='o', markersize=5, 
                     linestyle='None', alpha=0.85, capsize=3, 
                     label=label_str, zorder=5)
        
        color_idx += 1

    # Formatting the Plot
    title_metric = "S^2" if metric == 'S2' else "R_{ex}"
    ylabel_str = "Order Parameter ($S^2$)" if metric == 'S2' else "Chemical Exchange ($R_{ex}$)"
    
    plt.title(f"Protein Dynamics Profile (${title_metric}$) for Target: {title_prefix}", fontsize=16, fontweight='bold')
    plt.xlabel("Residue Number", fontsize=14)
    plt.ylabel(ylabel_str, fontsize=14)
    
    if all_residues or has_exp:
        min_res = min(all_residues) if all_residues else df_exp_clean.index.min()
        max_res = max(all_residues) if all_residues else df_exp_clean.index.max()
        plt.xlim(max(0, min_res - 5), max_res + 5)
        
    if metric == 'S2': plt.ylim(0.0, 1.05)
    else: plt.ylim(bottom=-0.1)
        
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=10)
    plt.tight_layout()
    
    out_pdf = os.path.join(out_dir, f"{title_prefix}_Combined_{metric}_Profile.pdf")
    plt.savefig(out_pdf, dpi=300)
    plt.close()

# ==========================================
# 4. FILE GATHERING AND PROCESSING LOOP
# ==========================================
def process_selected_files(file_paths, model_dir, out_dir, target_files=None):
    os.makedirs(out_dir, exist_ok=True)
    files_to_process = []
    temp_dirs = []

    for file in file_paths:
        if not os.path.isfile(file): continue
        if file.lower().endswith('.zip'):
            td = tempfile.mkdtemp(); temp_dirs.append(td)
            with zipfile.ZipFile(file, 'r') as z:
                z.extractall(td)
                for r, _, fs in os.walk(td): files_to_process.extend([os.path.join(r, f) for f in fs if f.endswith(('.str', '.txt', '.bmrb'))])
        elif file.lower().endswith(('.tar.gz', '.tgz', '.tar')):
            td = tempfile.mkdtemp(); temp_dirs.append(td)
            with tarfile.open(file, 'r:*') as t:
                t.extractall(td)
                for r, _, fs in os.walk(td): files_to_process.extend([os.path.join(r, f) for f in fs if f.endswith(('.str', '.txt', '.bmrb'))])
        else: files_to_process.append(file)

    if target_files:
        files_to_process = [f for f in files_to_process if any(t in os.path.basename(f) for t in target_files)]

    for file in files_to_process:
        entry_id, _ = extract_metadata(file)
        if entry_id == "Not Found": entry_id = os.path.basename(file).split('.')[0]

        print(f"\n⚙️ Processing File: {os.path.basename(file)} (ID: {entry_id})")
        
        dfs_relax, dfs_s2 = extract_multi_field_dynamics(file)
        if not dfs_relax:
            print("   ⚠️ No usable relaxation fields found. Skipping.")
            continue
            
        spindle_dfs_by_cond = {}
        for (field, cond), df_relax in dfs_relax.items():
            if float(field) not in CALIBRATIONS: continue
            models = load_ensemble(field, model_dir)
            if not models: continue
                
            df_pred = run_spindle(df_relax, field, models)
            if df_pred is not None: 
                df_pred['Field'] = field 
                df_pred['Condition'] = cond
                spindle_dfs_by_cond.setdefault(cond, []).append(df_pred)
            
        if not spindle_dfs_by_cond:
            print("   ⚠️ No valid SPINDLE predictions generated.")
            continue
            
        for cond, spindle_dfs in spindle_dfs_by_cond.items():
            print(f"   📊 Plotting Condition: '{cond}' ({len(spindle_dfs)} fields)...")
            
            df_exp_cond = dfs_s2.get(cond)
            if df_exp_cond is None and len(dfs_s2) == 1:
                df_exp_cond = list(dfs_s2.values())[0]
            elif df_exp_cond is None:
                df_exp_cond = pd.DataFrame()
            
            title_prefix = f"{entry_id}_{cond}"
            
            plot_combined_profile(df_exp_cond, spindle_dfs, title_prefix, out_dir, metric='S2')
            plot_combined_profile(df_exp_cond, spindle_dfs, title_prefix, out_dir, metric='Rex')
            
            try:
                combined_df = pd.concat(spindle_dfs, ignore_index=True)
                if not df_exp_cond.empty:
                    df_exp_reset = df_exp_cond.reset_index().rename(columns={'index': 'Residue', 'Residue_Number': 'Residue', 'S2': 'S2_exp', 'Rex': 'Rex_exp'})
                    merge_cols = ['Residue']
                    if 'S2_exp' in df_exp_reset.columns: merge_cols.append('S2_exp')
                    if 'Rex_exp' in df_exp_reset.columns: merge_cols.append('Rex_exp')
                    combined_df = pd.merge(combined_df, df_exp_reset[merge_cols], on='Residue', how='left')
                
                combined_df.to_csv(os.path.join(out_dir, f"{title_prefix}_Combined_Data.csv"), index=False)
            except Exception as e:
                print(f"   ⚠️ Could not save combined CSV for {cond}: {e}")
                
        print(f"   ✅ Saved plots and data to {out_dir}/")

    for td in temp_dirs: shutil.rmtree(td, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Combined S2 and Rex Profiles grouped by Condition.")
    parser.add_argument("-f", "--files", nargs='+', required=True, help="List of files or archives.")
    parser.add_argument("-t", "--targets", nargs='+', default=None, help="Optional: Specific filenames to extract.")
    parser.add_argument("-models", "--model_dir", required=True, help="Path to SPINDLE models.")
    parser.add_argument("-out", "--out_dir", default="results", help="Directory to save plots.")
    
    args = parser.parse_args()
    process_selected_files(args.files, args.model_dir, args.out_dir, target_files=args.targets)

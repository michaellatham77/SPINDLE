#!/usr/bin/env python3

import os
import sys
import argparse
import tarfile
import zipfile
import shutil
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import pearsonr
from spindle_calibrations import CALIBRATIONS, MULTIPLIERS

# Force CPU for inference (stable for sequential file processing)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

AMBIGUITY_THRESHOLD_S2 = 0.05
AMBIGUITY_THRESHOLD_REX = 2.0

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
    s2_data = {}
    current_field = "Unknown"
    current_save_cond = "Cond_1"
    in_loop = False
    in_data = False
    current_loop_tags = []
    
    current_unit = "s-1" 
    file_taue_unit = None  
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        if stripped.startswith('save_'):
            current_unit = "s-1"
            current_save_cond = "Cond_1" # Reset condition for new save frame
            
        if stripped.startswith('loop_'):
            current_loop_tags = []
            in_loop = True
            in_data = False
            continue
            
        # --- NEW: CAPTURE METADATA OUTSIDE THE LOOP ---
        if not in_loop and stripped.startswith('_'):
            parts = stripped.split(maxsplit=1)
            if len(parts) > 1 and parts[1] != '.':
                tag = parts[0]
                val = parts[1].strip('\'"')
                
                if 'Spectrometer_frequency_1H' in tag:
                    current_field = snap_field(val)
                elif '.Val_units' in tag or '.T2_val_units' in tag or '.T1_val_units' in tag:
                    current_unit = val.lower()
                elif '.Tau_e_val_units' in tag:
                    file_taue_unit = val.lower()
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
                
                # --- FIELD AND CONDITION EXTRACTION ---
                field_t1 = snap_field(row_dict.get('_T1.Spectrometer_frequency_1H', current_field)) if row_dict.get('_T1.Spectrometer_frequency_1H', '.') != '.' else current_field
                field_t2 = snap_field(row_dict.get('_T2.Spectrometer_frequency_1H', current_field)) if row_dict.get('_T2.Spectrometer_frequency_1H', '.') != '.' else current_field
                field_noe = snap_field(row_dict.get('_Heteronucl_NOE.Spectrometer_frequency_1H', current_field)) if row_dict.get('_Heteronucl_NOE.Spectrometer_frequency_1H', '.') != '.' else current_field
                
                # Now passing the memorized global condition as a fallback!
                cond_t1 = get_cond(row_dict, "T1", current_save_cond)
                cond_t2 = get_cond(row_dict, "T2", current_save_cond)
                cond_noe = get_cond(row_dict, "Heteronucl_NOE", current_save_cond)
                
                # Extract Relaxation Data Using the (Field, Condition) Key Pair
                if '_T1.Val' in row_dict and '_T1.Seq_ID' in row_dict and row_dict['_T1.Val'] != '.':
                    val = float(row_dict['_T1.Val'])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        seq_id = int(row_dict['_T1.Seq_ID'])
                        relax_data.setdefault((field_t1, cond_t1), {}).setdefault(seq_id, {})['R1'] = val
                    
                elif '_T2.T2_val' in row_dict and '_T2.Seq_ID' in row_dict and row_dict['_T2.T2_val'] != '.':
                    val = float(row_dict['_T2.T2_val'])
                    if val != 0.0:
                        if current_unit in ['s', 'sec', 'seconds']: val = 1.0 / val
                        elif current_unit in ['ms', 'msec', 'milliseconds']: val = 1000.0 / val
                        elif current_unit in ['ms-1', 'ms^-1']: val = val * 1000.0
                        seq_id = int(row_dict['_T2.Seq_ID'])
                        relax_data.setdefault((field_t2, cond_t2), {}).setdefault(seq_id, {})['R2'] = val
                    
                elif '_Heteronucl_NOE.Val' in row_dict and '_Heteronucl_NOE.Seq_ID_1' in row_dict and row_dict['_Heteronucl_NOE.Val'] != '.':
                    val = float(row_dict['_Heteronucl_NOE.Val'])
                    seq_id = int(row_dict['_Heteronucl_NOE.Seq_ID_1'])
                    relax_data.setdefault((field_noe, cond_noe), {}).setdefault(seq_id, {})['NOE'] = val
                
                # Extract S2, TauE, and Rex
                if '_Order_param.Order_param_val' in row_dict and '_Order_param.Seq_ID' in row_dict and row_dict['_Order_param.Order_param_val'] != '.':
                    seq_id = int(row_dict['_Order_param.Seq_ID'])
                    val_s2 = float(row_dict['_Order_param.Order_param_val'])
                    s2_data.setdefault(seq_id, {})['S2'] = val_s2
                    
                    if '_Order_param.Tau_e_val' in row_dict and row_dict['_Order_param.Tau_e_val'] != '.':
                        val_te = float(row_dict['_Order_param.Tau_e_val'])
                        s2_data.setdefault(seq_id, {})['Tau_e'] = val_te

                    if '_Order_param.Rex_val' in row_dict and row_dict['_Order_param.Rex_val'] != '.':
                        val_rex = float(row_dict['_Order_param.Rex_val'])
                        s2_data.setdefault(seq_id, {})['Rex'] = val_rex

    dfs_relax = {}
    for (field, cond), data_dict in relax_data.items():
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        if not df.empty:
            df.index.name = 'Residue'
            df.sort_index(inplace=True)
            cols = [c for c in ['R1', 'R2', 'NOE'] if c in df.columns]
            dfs_relax[(field, cond)] = df[cols]
            
    # --- FOOLPROOF RELAXATION FALLBACK HEURISTIC ---
    for (field, cond), df in dfs_relax.items():
        median_r1 = df['R1'].median() if 'R1' in df.columns else None
        median_r2 = df['R2'].median() if 'R2' in df.columns else None
        
        if median_r2 is not None and median_r2 < 1.0:
            print(f"   ⚙️  Auto-Correction [{field} MHz | {cond}]: Median R2 is {median_r2:.3f}. Undeniably SECONDS. Converting to rates...")
            df['R2'] = 1.0 / df['R2']
            if 'R1' in df.columns: df['R1'] = 1.0 / df['R1']
            
        elif median_r1 is not None and median_r1 > 50.0:
            print(f"   ⚙️  Auto-Correction [{field} MHz | {cond}]: Median R1 is {median_r1:.0f}. Undeniably MILLISECONDS. Converting to rates...")
            if 'R1' in df.columns: df['R1'] = 1000.0 / df['R1']
            if 'R2' in df.columns: df['R2'] = 1000.0 / df['R2']

    df_s2 = pd.DataFrame.from_dict(s2_data, orient='index')
    if not df_s2.empty:
        df_s2.index.name = 'Residue_Number'
        df_s2.sort_index(inplace=True)
        cols = [c for c in ['S2', 'Tau_e', 'Rex'] if c in df_s2.columns]
        df_s2 = df_s2[cols]
        
        # --- ABSOLUTE MAGNITUDE ENFORCER FOR TAU_E ---
        if 'Tau_e' in df_s2.columns:
            df_s2['Tau_e'] = df_s2['Tau_e'].replace(0.0, np.nan)
            df_s2['Tau_e'] = pd.to_numeric(df_s2['Tau_e'], errors='coerce')
            
            valid_taue = df_s2['Tau_e'].dropna()
            if not valid_taue.empty:
                median_taue = valid_taue.median()
                
                if median_taue < 1e-6:
                    print(f"   ⚙️  Auto-Correction: Raw Tau_e median is {median_taue:.2e}. Undeniably SECONDS. Overriding metadata and converting to ps...")
                    df_s2['Tau_e'] = df_s2['Tau_e'] * 1e12
                elif median_taue < 5.0:
                    if file_taue_unit in ['ps', 'psec', 'picoseconds']:
                        print(f"   ⚙️  Auto-Correction: Raw Tau_e median is {median_taue:.2f}, explicitly tagged PICOSECONDS. Leaving as is.")
                    else:
                        print(f"   ⚙️  Auto-Correction: Raw Tau_e median is {median_taue:.3f}. Assuming NANOSECONDS. Converting to ps...")
                        df_s2['Tau_e'] = df_s2['Tau_e'] * 1000.0
                else:
                    if file_taue_unit in ['ns', 'nsec', 'nanoseconds'] and median_taue < 100.0:
                        print(f"   ⚙️  Auto-Correction: Raw Tau_e median is {median_taue:.1f}, explicitly tagged NANOSECONDS. Converting to ps...")
                        df_s2['Tau_e'] = df_s2['Tau_e'] * 1000.0
                    else:
                        print(f"   ⚙️  Auto-Correction: Raw Tau_e median is {median_taue:.1f}. Assuming already PICOSECONDS.")

    return dfs_relax, df_s2

# ==========================================
# 2. SPINDLE INFERENCE FUNCTIONS
# ==========================================
def load_ensemble(field_mhz, base_model_dir):
    model_dir = os.path.join(base_model_dir, str(int(float(field_mhz))))
    if not os.path.exists(model_dir): 
        return None
    models = []
    for m_file in sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')]):
        try: 
            models.append(tf.keras.models.load_model(os.path.join(model_dir, m_file), compile=False))
        except Exception: 
            pass
    return models

def apply_calibration(pred_mean, pred_err, calib, mult):
    s2_corr = np.clip((pred_mean[:, 1] * calib['S2']['slope']) + calib['S2']['intercept'], 0.0, 1.0)
    s2_corr_err = pred_err[:,1] / calib['S2']['slope'] * mult['S2']['mult']
    
    tauE_ps = pred_mean[:, 0] * 1000.0
    tauE_corr_ps = np.maximum((tauE_ps * calib['TAUE']['slope']) + calib['TAUE']['intercept'], 0)
    tauE_corr_err = pred_err[:,0] * 1000.0 / calib['TAUE']['slope'] * mult['TAUE']['mult']
    
    rex = pred_mean[:, 2]
    rex_corr = np.maximum((calib['REX']['a'] * rex**2) + (calib['REX']['b'] * rex) + calib['REX']['c'], 0.0)
    rex_deriv = np.where(np.abs(2 * calib['REX']['a'] * rex + calib['REX']['b']) == 0, 1e-6, np.abs(2 * calib['REX']['a'] * rex + calib['REX']['b']))
    rex_corr_err = pred_err[:,2] / rex_deriv * mult['REX']['mult']
    
    return np.stack([tauE_corr_ps, s2_corr, rex_corr], axis=1), np.stack([tauE_corr_err, s2_corr_err, rex_corr], axis=1)

def run_spindle(df_relax, field, models):
    required_cols = ['R1', 'R2', 'NOE']
    missing_cols = [c for c in required_cols if c not in df_relax.columns]
    if missing_cols:
        print(f"   ⚠️ Field {field} MHz is missing {', '.join(missing_cols)}. SPINDLE requires all 3. Skipping.")
        return None
        
    df = df_relax.dropna(subset=required_cols).copy()
    if df.empty: 
        print(f"   ⚠️ Field {field} MHz has no residues with all three measurements. Skipping.")
        return None
    
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
    tauc_std = np.std(tauc_preds) / calib['TAUC']['slope'] * mult['TAUC']['mult']
    
    local_mean = np.mean(local_preds, axis=0)
    local_err = np.std(local_preds, axis=0)
    loc_corr_mean, loc_corr_err = apply_calibration(local_mean, local_err, calib, mult)
    
    print(f"Global tau_c: {tauc_mean:.2f} ± {tauc_std:.2f} ns")

    results = []
    for i, res in enumerate(residues):
        quality = "Ambiguous" if (loc_corr_err[i, 1] > AMBIGUITY_THRESHOLD_S2 or loc_corr_err[i, 2] > AMBIGUITY_THRESHOLD_REX) else "Good"
        results.append({
            'Residue': res,
            'S2_pred': loc_corr_mean[i, 1], 'S2_err': loc_corr_err[i, 1],
            'Tau_e_pred': loc_corr_mean[i, 0], 'tauE_err_ps': loc_corr_err[i, 0],
            'Rex_pred': loc_corr_mean[i, 2], 'Rex_err': loc_corr_err[i, 2],
            'Quality': quality, 'tauC_global': tauc_mean
        })
    return pd.DataFrame(results)

# ==========================================
# 3. COMPARISON & PLOTTING FUNCTIONS
# ==========================================
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float('nan'), float('nan'), float('nan')
    r_val, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r_val, rmse, mae

def compare_and_plot(df_exp, df_pred, entry_id, out_dir, suffix="Averaged"):
    if 'Residue_Number' not in df_exp.columns:
        df_exp = df_exp.reset_index()
        
    df_exp = df_exp.rename(columns={'S2': 'S2_exp', 'Tau_e': 'Tau_e_exp', 'Rex': 'Rex_exp'})
    df_merged = pd.merge(df_exp, df_pred, left_on='Residue_Number', right_on='Residue')

    if df_merged.empty: 
        return None

    stats = {'Entry_ID': entry_id, 'Field_Condition': suffix,
             'TauC_Pred_ns': np.nan, 
             'N_S2': 0, 'S2_R': np.nan, 'S2_RMSE': np.nan, 'S2_MAE': np.nan,
             'N_TauE': 0, 'TauE_R': np.nan, 'TauE_RMSE': np.nan, 'TauE_MAE': np.nan,
             'N_Rex': 0, 'Rex_R': np.nan, 'Rex_RMSE': np.nan, 'Rex_MAE': np.nan}
             
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='gray')
    
    if 'tauC_global' in df_pred.columns:
        mean_tauc = df_pred['tauC_global'].mean()
        stats['TauC_Pred_ns'] = round(mean_tauc, 3)
        fig.suptitle(f"Entry {entry_id} ({suffix}) | Predicted Global $\\tau_c$: {mean_tauc:.2f} ns", fontsize=16, fontweight='bold')

    # S2 Analysis
    df_s2 = df_merged.dropna(subset=['S2_exp', 'S2_pred'])
    if len(df_s2) > 1:
        s2_exp = df_s2['S2_exp'].values.astype(float)
        s2_pred = df_s2['S2_pred'].values.astype(float)
        r, rmse, mae = calculate_metrics(s2_exp, s2_pred)
        stats.update({'N_S2': len(df_s2), 'S2_R': r, 'S2_RMSE': rmse, 'S2_MAE': mae})
        ax1.scatter(s2_exp, s2_pred, alpha=0.7, color='dodgerblue', edgecolors='k', s=60)
        m_val = min(s2_exp.min(), s2_pred.min()) - 0.05
        x_val = max(s2_exp.max(), s2_pred.max()) + 0.05
        ax1.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
        ax1.text(0.05, 0.95, f"N = {len(df_s2)}\n$r$ = {r:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}", transform=ax1.transAxes, verticalalignment='top', bbox=props)
    ax1.set_title('Order Parameters ($S^2$)')
    ax1.set_xlabel('Experimental S²')
    ax1.set_ylabel('Predicted S²')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # TauE Analysis
    if 'Tau_e_exp' in df_merged.columns and 'Tau_e_pred' in df_merged.columns:
        df_taue = df_merged.dropna(subset=['Tau_e_exp', 'Tau_e_pred'])
        if len(df_taue) > 1:
            te_exp = df_taue['Tau_e_exp'].values.astype(float)
            te_pred = df_taue['Tau_e_pred'].values.astype(float)
            r, rmse, mae = calculate_metrics(te_exp, te_pred)
            stats.update({'N_TauE': len(df_taue), 'TauE_R': r, 'TauE_RMSE': rmse, 'TauE_MAE': mae})
            ax2.scatter(te_exp, te_pred, alpha=0.7, color='darkorange', edgecolors='k', s=60)
            m_val = min(te_exp.min(), te_pred.min()) * 0.9
            x_val = max(te_exp.max(), te_pred.max()) * 1.1
            ax2.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
            ax2.text(0.05, 0.95, f"N = {len(df_taue)}\n$r$ = {r:.3f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}", transform=ax2.transAxes, verticalalignment='top', bbox=props)
        else:
            ax2.text(0.5, 0.5, "Insufficient overlapping\n$\\tau_e$ data to plot.", ha='center', va='center', fontsize=14, color='gray')
    else:
        ax2.text(0.5, 0.5, "No Experimental\n$\\tau_e$ Data", ha='center', va='center', fontsize=14, color='gray')
    ax2.set_title('Internal Motion Timescales ($\\tau_e$)')
    ax2.set_xlabel('Experimental $\\tau_e$ (ps)')
    ax2.set_ylabel('Predicted $\\tau_e$ (ps)')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Rex Analysis
    if 'Rex_exp' in df_merged.columns and 'Rex_pred' in df_merged.columns:
        df_rex = df_merged.dropna(subset=['Rex_exp', 'Rex_pred'])
        if len(df_rex) > 1:
            rex_exp = df_rex['Rex_exp'].values.astype(float)
            rex_pred = df_rex['Rex_pred'].values.astype(float)
            r, rmse, mae = calculate_metrics(rex_exp, rex_pred)
            stats.update({'N_Rex': len(df_rex), 'Rex_R': r, 'Rex_RMSE': rmse, 'Rex_MAE': mae})
            ax3.scatter(rex_exp, rex_pred, alpha=0.7, color='mediumseagreen', edgecolors='k', s=60)
            m_val = min(rex_exp.min(), rex_pred.min()) - 0.5
            x_val = max(rex_exp.max(), rex_pred.max()) + 0.5
            ax3.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
            ax3.text(0.05, 0.95, f"N = {len(df_rex)}\n$r$ = {r:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}", transform=ax3.transAxes, verticalalignment='top', bbox=props)
        else:
            ax3.text(0.5, 0.5, "Insufficient overlapping\n$R_{ex}$ data to plot.", ha='center', va='center', fontsize=14, color='gray')
    else:
        ax3.text(0.5, 0.5, "No Experimental\n$R_{ex}$ Data", ha='center', va='center', fontsize=14, color='gray')
    
    ax3.set_title('Chemical Exchange ($R_{ex}$)')
    ax3.set_xlabel('Experimental $R_{ex}$')
    ax3.set_ylabel('Predicted $R_{ex}$')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_name = f"{entry_id}_{suffix}"
    plt.savefig(os.path.join(out_dir, f"{out_name}_comparison.pdf"), dpi=300)
    plt.close()
    
    df_merged.to_csv(os.path.join(out_dir, f"{out_name}_combined_data.csv"), index=False)
    return stats


def plot_global_comparison(s2_exp, s2_pred, taue_exp, taue_pred, rex_exp, rex_pred, all_stats, out_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Global SPINDLE Predictions vs NMR-STAR Database (All Fields)", fontsize=18, fontweight='bold')
    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='gray')
    
    # Global S2
    if len(s2_exp) > 1:
        r, rmse, mae = calculate_metrics(s2_exp, s2_pred)
        ax1.scatter(s2_exp, s2_pred, alpha=0.5, color='dodgerblue', edgecolors='none', s=40)
        m_val = min(s2_exp.min(), s2_pred.min()) - 0.05
        x_val = max(s2_exp.max(), s2_pred.max()) + 0.05
        ax1.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
        ax1.text(0.05, 0.95, f"Total Data Points = {len(s2_exp)}\n$r$ = {r:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}", transform=ax1.transAxes, verticalalignment='top', bbox=props)
    ax1.set_title('GLOBAL Order Parameters ($S^2$)')
    ax1.set_xlabel('Experimental S²')
    ax1.set_ylabel('Predicted S²')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Global TauE
    if len(taue_exp) > 1:
        r, rmse, mae = calculate_metrics(taue_exp, taue_pred)
        ax2.scatter(taue_exp, taue_pred, alpha=0.5, color='darkorange', edgecolors='none', s=40)
        m_val = min(taue_exp.min(), taue_pred.min()) * 0.9
        x_val = max(taue_exp.max(), taue_pred.max()) * 1.1
        ax2.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
        ax2.text(0.05, 0.95, f"Total Data Points = {len(taue_exp)}\n$r$ = {r:.3f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}", transform=ax2.transAxes, verticalalignment='top', bbox=props)
        
        # TauE Inset
        axins = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
        axins.scatter(taue_exp, taue_pred, alpha=0.5, color='darkorange', edgecolors='none', s=15)
        axins.plot([0, 2000], [0, 2000], 'r--', linewidth=1.5)
        axins.set_xlim(0, 2000)
        axins.set_ylim(0, 2000)
        axins.grid(True, linestyle=':', alpha=0.6)
        axins.tick_params(labelsize=8)
        ax2.indicate_inset_zoom(axins, edgecolor="black", alpha=0.3)

    ax2.set_title('GLOBAL Internal Motion ($\\tau_e$)')
    ax2.set_xlabel('Experimental $\\tau_e$ (ps)')
    ax2.set_ylabel('Predicted $\\tau_e$ (ps)')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Global Rex
    if len(rex_exp) > 1:
        r, rmse, mae = calculate_metrics(rex_exp, rex_pred)
        ax3.scatter(rex_exp, rex_pred, alpha=0.5, color='mediumseagreen', edgecolors='none', s=40)
        m_val = min(rex_exp.min(), rex_pred.min()) - 0.5
        x_val = max(rex_exp.max(), rex_pred.max()) + 0.5
        ax3.plot([m_val, x_val], [m_val, x_val], 'r--', linewidth=2)
        ax3.text(0.05, 0.95, f"Total Data Points = {len(rex_exp)}\n$r$ = {r:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}", transform=ax3.transAxes, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.5, 0.5, "Insufficient overlapping\n$R_{ex}$ data to plot.", ha='center', va='center', fontsize=14, color='gray')
    ax3.set_title('GLOBAL Chemical Exchange ($R_{ex}$)')
    ax3.set_xlabel('Experimental $R_{ex}$')
    ax3.set_ylabel('Predicted $R_{ex}$')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    global_pdf = os.path.join(out_dir, "GLOBAL_COMPARISON.pdf")
    plt.savefig(global_pdf, dpi=300)
    plt.close()
    print(f"📊 Saved Master Global Plot to: {global_pdf}")
    
    # Generate the Histogram
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        if 'S2_R' in df_stats.columns:
            avg_stats = df_stats[df_stats['Field_Condition'] == 'Averaged_All_Conditions']
            valid_s2_r = avg_stats['S2_R'].dropna()
            if len(valid_s2_r) > 0:
                plt.figure(figsize=(8, 6))
                plt.hist(valid_s2_r, bins=10, color='dodgerblue', edgecolor='black', alpha=0.8)
                plt.title('Distribution of S² Pearson r Across All Entries (Consensus Only)')
                plt.xlabel('Pearson r')
                plt.ylabel('Frequency')
                plt.grid(axis='y', linestyle=':', alpha=0.7)
                hist_pdf = os.path.join(out_dir, 'GLOBAL_S2_Pearson_Histogram.pdf')
                plt.savefig(hist_pdf, dpi=300)
                plt.close()
                print(f"📊 Saved Pearson r Histogram Plot to: {hist_pdf}")

# ==========================================
# 4. MAIN PIPELINE LOOP
# ==========================================
def process_archive(archive_path, model_dir, out_dir, ignore_entries):
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"📦 Extracting {archive_path}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as z: 
            z.extractall(temp_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as t: 
            t.extractall(temp_dir)
    else:
        print("❌ Unsupported archive format. Use .zip or .tar.gz")
        sys.exit(1)

    nmr_files = glob.glob(os.path.join(temp_dir, '**', '*.str'), recursive=True) + glob.glob(os.path.join(temp_dir, '**', '*.txt'), recursive=True)
    print(f"🔍 Found {len(nmr_files)} potential NMR-STAR files.")

    all_stats = []
    doi_mapping = []
    
    global_exp_s2, global_pred_s2 = [], []
    global_exp_taue, global_pred_taue = [], []
    global_exp_rex, global_pred_rex = [], []

    for file in nmr_files:
        entry_id, doi = extract_metadata(file)
        
        if entry_id in ignore_entries:
            print(f"\n⏭️  Skipping Entry: {entry_id} (Found in Ignore List)")
            continue
            
        print(f"\n⚙️  Processing Entry: {entry_id} (DOI: {doi})")
        doi_mapping.append({'Entry_ID': entry_id, 'DOI': doi})
        
        dfs_relax, df_s2 = extract_multi_field_dynamics(file)
        if df_s2.empty:
            print("   ⚠️ No experimental S2 data found. Skipping comparison.")
            continue
            
        spindle_dfs = []
        for (field, cond), df_relax in dfs_relax.items():
            if float(field) not in CALIBRATIONS:
                print(f"   ⚠️ Field {field} MHz (Cond: {cond}) not supported by SPINDLE. Skipping.")
                continue
                
            models = load_ensemble(field, model_dir)
            if not models:
                print(f"   ⚠️ Models for {field} MHz not found in {model_dir}. Skipping.")
                continue
                
            df_pred = run_spindle(df_relax, field, models)
            if df_pred is not None: 
                df_pred['Field'] = field 
                df_pred['Condition'] = cond
                spindle_dfs.append(df_pred)
            
        if not spindle_dfs:
            print("   ⚠️ No valid SPINDLE predictions generated for this entry.")
            continue
            
        for df_field in spindle_dfs:
            current_field = float(df_field['Field'].iloc[0])
            current_cond = df_field['Condition'].iloc[0]
            suffix_str = f"{current_field:g}MHz_{current_cond}"
            print(f"   📊 Plotting individual field: {current_field:g} MHz ({current_cond})")
            
            field_stats = compare_and_plot(df_s2, df_field, entry_id, out_dir, suffix=suffix_str)
            if field_stats:
                all_stats.append(field_stats)
            
        df_concat = pd.concat(spindle_dfs)
        df_concat_clean = df_concat.drop(columns=['Field', 'Condition'], errors='ignore')
        
        numeric_cols = [c for c in df_concat_clean.select_dtypes(include='number').columns if c != 'Residue']
        non_numeric = [c for c in df_concat_clean.select_dtypes(exclude='number').columns if c != 'Residue']
        
        agg_dict = {col: 'mean' for col in numeric_cols}
        agg_dict.update({col: 'first' for col in non_numeric})
        df_avg_pred = df_concat_clean.groupby('Residue').agg(agg_dict).reset_index()
        
        print("   📊 Plotting averaged consensus fields...")
        
        stats = compare_and_plot(df_s2, df_avg_pred, entry_id, out_dir, suffix="Averaged_All_Conditions")
        if stats:
            all_stats.append(stats)
            print(f"   ✅ Processed! S2 Pearson r (Averaged): {stats['S2_R']:.3f}")
            
            df_s2_flat = df_s2.reset_index() if 'Residue_Number' not in df_s2.columns else df_s2.copy()
            merged_all_fields = pd.merge(df_s2_flat.rename(columns={'S2':'S2_exp', 'Tau_e':'Tau_e_exp', 'Rex':'Rex_exp'}), df_concat, left_on='Residue_Number', right_on='Residue')
            
            df_glob_s2 = merged_all_fields.dropna(subset=['S2_exp', 'S2_pred'])
            global_exp_s2.extend(df_glob_s2['S2_exp'].astype(float).tolist())
            global_pred_s2.extend(df_glob_s2['S2_pred'].astype(float).tolist())
            
            if 'Tau_e_exp' in merged_all_fields.columns:
                df_glob_te = merged_all_fields.dropna(subset=['Tau_e_exp', 'Tau_e_pred'])
                global_exp_taue.extend(df_glob_te['Tau_e_exp'].astype(float).tolist())
                global_pred_taue.extend(df_glob_te['Tau_e_pred'].astype(float).tolist())
                
            if 'Rex_exp' in merged_all_fields.columns and 'Rex_pred' in merged_all_fields.columns:
                df_glob_rex = merged_all_fields.dropna(subset=['Rex_exp', 'Rex_pred'])
                global_exp_rex.extend(df_glob_rex['Rex_exp'].astype(float).tolist())
                global_pred_rex.extend(df_glob_rex['Rex_pred'].astype(float).tolist())

    shutil.rmtree(temp_dir)

    if not all_stats:
        print("\n❌ Pipeline finished, but no valid comparisons were generated.")
        sys.exit(0)
        
    print("\n==============================================")
    print("🎉 PIPELINE COMPLETE! CALCULATING GLOBAL STATS")
    print("==============================================")
    
    r_s2, rmse_s2, mae_s2 = calculate_metrics(np.array(global_exp_s2, dtype=float), np.array(global_pred_s2, dtype=float))
    print(f"Total S2 Data Points Analyzed : {len(global_exp_s2)}")
    print(f"Global S2 Pearson r           : {r_s2:.3f}")
    print(f"Global S2 RMSE                : {rmse_s2:.3f}")
    print(f"Global S2 MAE                 : {mae_s2:.3f}")
    
    if len(global_exp_taue) > 1:
        r_te, rmse_te, mae_te = calculate_metrics(np.array(global_exp_taue, dtype=float), np.array(global_pred_taue, dtype=float))
        print(f"\nTotal TauE Data Points Analyzed: {len(global_exp_taue)}")
        print(f"Global TauE Pearson r          : {r_te:.3f}")
        print(f"Global TauE RMSE               : {rmse_te:.1f} ps")
        print(f"Global TauE MAE                : {mae_te:.1f} ps")
        
    if len(global_exp_rex) > 1:
        r_rex, rmse_rex, mae_rex = calculate_metrics(np.array(global_exp_rex, dtype=float), np.array(global_pred_rex, dtype=float))
        print(f"\nTotal Rex Data Points Analyzed: {len(global_exp_rex)}")
        print(f"Global Rex Pearson r          : {r_rex:.3f}")
        print(f"Global Rex RMSE               : {rmse_rex:.3f}")
        print(f"Global Rex MAE                : {mae_rex:.3f}")
        
    print("==============================================\n")

    plot_global_comparison(
        np.array(global_exp_s2, dtype=float), np.array(global_pred_s2, dtype=float), 
        np.array(global_exp_taue, dtype=float), np.array(global_pred_taue, dtype=float), 
        np.array(global_exp_rex, dtype=float), np.array(global_pred_rex, dtype=float), 
        all_stats, out_dir
    )

    df_stats = pd.DataFrame(all_stats)
    stats_file = os.path.join(out_dir, "GLOBAL_STATISTICS_SUMMARY.csv")
    df_stats.to_csv(stats_file, index=False)
    print(f"📄 Saved per-entry statistics table to: {stats_file}")
    
    df_doi = pd.DataFrame(doi_mapping)
    doi_file = os.path.join(out_dir, "ENTRY_DOI_MAPPING.csv")
    df_doi.to_csv(doi_file, index=False)
    print(f"📄 Saved DOI mapping table to: {doi_file}")
    print(f"📂 All individual plots and CSVs are located in: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Massive NMR-STAR Processing Pipeline for SPINDLE.")
    parser.add_argument("-in", "--input", required=True, help="Path to the compressed archive (.zip or .tar.gz) containing NMR-STAR files.")
    parser.add_argument("-models", "--model_dir", required=True, help="Path to the base directory containing SPINDLE models (e.g., ./models)")
    parser.add_argument("-out", "--out_dir", default="pipeline_results", help="Directory to save all plots and tables (default: pipeline_results)")
    parser.add_argument("-ignore", "--ignore_entries", nargs='*', default=[], help="List of Entry IDs to skip, separated by spaces (e.g., -ignore 50285 4970).")
    
    args = parser.parse_args()
    process_archive(args.input, args.model_dir, args.out_dir, args.ignore_entries)

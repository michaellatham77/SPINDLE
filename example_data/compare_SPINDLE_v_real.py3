#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import os

def calculate_metrics(y_true, y_pred):
    """Calculates Pearson r, RMSE, and MAE. Returns NaNs if not enough data."""
    if len(y_true) < 2:
        return float('nan'), float('nan'), float('nan')
    r_val, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r_val, rmse, mae

def compare_dynamics_data(truth_file, pred_file, output_plot, file_format):
    # 1. Load the data
    print(f"Loading ground truth data from: {truth_file}")
    df_truth = pd.read_csv(truth_file)
    
    print(f"Loading predicted data from: {pred_file}")
    df_pred = pd.read_csv(pred_file)
    
    # 2. Merge datasets based on Residue Number
    df_merged = pd.merge(df_truth, df_pred, on='Residue')
    
    # --- S2 ANALYSIS ---
    df_s2 = df_merged.dropna(subset=['True_S2', 'S2'])
    s2_truth = df_s2['True_S2'].values
    s2_pred = df_s2['S2'].values
    r_s2, rmse_s2, mae_s2 = calculate_metrics(s2_truth, s2_pred)
    
    print("\n--- S² Statistical Results ---")
    print(f"Overlapping S² residues: {len(df_s2)}")
    print(f"Pearson Correlation (r): {r_s2:.3f}")
    print(f"RMSE: {rmse_s2:.3f}")
    print(f"MAE:  {mae_s2:.3f}")

    # --- TAU_E ANALYSIS ---
    df_taue = df_merged.dropna(subset=['True_tauE_ps', 'tauE_ps'])
    taue_truth = df_taue['True_tauE_ps'].values
    taue_pred = df_taue['tauE_ps'].values
    r_taue, rmse_taue, mae_taue = calculate_metrics(taue_truth, taue_pred)
    
    print("\n--- Tau_e Statistical Results ---")
    print(f"Overlapping Tau_e residues: {len(df_taue)}")
    if len(df_taue) > 1:
        print(f"Pearson Correlation (r): {r_taue:.3f}")
        print(f"RMSE (ps): {rmse_taue:.1f}")
        print(f"MAE (ps):  {mae_taue:.1f}")
    else:
        print("Not enough overlapping Tau_e points to calculate statistics.")
        
    # --- REX ANALYSIS ---
    df_rex = df_merged.dropna(subset=['True_Rex_s1', 'Rex'])
    rex_truth = df_rex['True_Rex_s1'].values
    rex_pred = df_rex['Rex'].values
    r_rex, rmse_rex, mae_rex = calculate_metrics(rex_truth, rex_pred)
    
    print("\n--- R_ex Statistical Results ---")
    print(f"Overlapping R_ex residues: {len(df_rex)}")
    if len(df_rex) > 1:
        print(f"Pearson Correlation (r): {r_rex:.3f}")
        print(f"RMSE (s⁻¹): {rmse_rex:.3f}")
        print(f"MAE (s⁻¹):  {mae_rex:.3f}")
    else:
        print("Not enough overlapping R_ex points to calculate statistics.")
        
    # --- GLOBAL TAU_C ANALYSIS ---
    print("\n--- Global Tau_c Comparison ---")
    # Using the first row since tauC is a global value for the entire protein
    if 'True_tauC_ns' in df_merged.columns and 'tauC_global' in df_merged.columns:
        tauc_truth = df_merged['True_tauC_ns'].iloc[0]
        tauc_pred = df_merged['tauC_global'].iloc[0]
        tauc_diff = abs(tauc_truth - tauc_pred)
        print(f"Ground Truth Tau_c: {tauc_truth:.2f} ns")
        print(f"SPINDLE Predicted:  {tauc_pred:.2f} ns")
        print(f"Absolute Error:     {tauc_diff:.2f} ns")
    else:
        print("Tau_c columns not found in one or both files. Check column names.")
    print("---------------------------------\n")

    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='gray')
    
    # 1. Plot S2
    if len(df_s2) > 1:
        ax1.scatter(s2_truth, s2_pred, alpha=0.7, color='dodgerblue', edgecolors='k', s=60, label='Residues')
        min_s2 = min(s2_truth.min(), s2_pred.min()) - 0.05
        max_s2 = max(s2_truth.max(), s2_pred.max()) + 0.05
        ax1.plot([min_s2, max_s2], [min_s2, max_s2], 'r--', linewidth=2, label='Perfect Agreement (y=x)')
        
        ax1.set_xlabel('Ground Truth S²', fontsize=12)
        ax1.set_ylabel('Predicted S² (SPINDLE)', fontsize=12)
        ax1.set_title('Order Parameters (S²)', fontsize=14)
        ax1.legend(loc='lower right')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        stats_text_s2 = '\n'.join((
            f'N = {len(df_s2)}',
            f'Pearson $r$ = {r_s2:.3f}',
            f'RMSE = {rmse_s2:.3f}',
            f'MAE = {mae_s2:.3f}'))
        ax1.text(0.05, 0.95, stats_text_s2, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    # 2. Plot Tau_e
    if len(df_taue) > 1:
        ax2.scatter(taue_truth, taue_pred, alpha=0.7, color='darkorange', edgecolors='k', s=60, label='Residues')
        min_taue = min(taue_truth.min(), taue_pred.min()) * 0.9 - 1.0
        max_taue = max(taue_truth.max(), taue_pred.max()) * 1.1 + 1.0
        ax2.plot([min_taue, max_taue], [min_taue, max_taue], 'r--', linewidth=2, label='Perfect Agreement (y=x)')
        
        ax2.set_xlabel('Ground Truth Tau_e (ps)', fontsize=12)
        ax2.set_ylabel('Predicted Tau_e (ps)', fontsize=12)
        ax2.set_title('Internal Motion Timescales (Tau_e)', fontsize=14)
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        stats_text_taue = '\n'.join((
            f'N = {len(df_taue)}',
            f'Pearson $r$ = {r_taue:.3f}',
            f'RMSE = {rmse_taue:.1f} ps',
            f'MAE = {mae_taue:.1f} ps'))
        ax2.text(0.05, 0.95, stats_text_taue, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    else:
        ax2.text(0.5, 0.5, "Insufficient overlapping\nTau_e data to plot.", 
                 ha='center', va='center', fontsize=14, color='gray')
        ax2.set_title('Internal Motion Timescales (Tau_e)', fontsize=14)
        
    # 3. Plot R_ex
    if len(df_rex) > 1:
        ax3.scatter(rex_truth, rex_pred, alpha=0.7, color='forestgreen', edgecolors='k', s=60, label='Residues')
        min_rex = min(rex_truth.min(), rex_pred.min()) - 0.5
        max_rex = max(rex_truth.max(), rex_pred.max()) + 0.5
        ax3.plot([min_rex, max_rex], [min_rex, max_rex], 'r--', linewidth=2, label='Perfect Agreement (y=x)')
        
        ax3.set_xlabel('Ground Truth R_ex (s⁻¹)', fontsize=12)
        ax3.set_ylabel('Predicted R_ex (s⁻¹)', fontsize=12)
        ax3.set_title('Chemical Exchange (R_ex)', fontsize=14)
        ax3.legend(loc='lower right')
        ax3.grid(True, linestyle=':', alpha=0.6)
        
        stats_text_rex = '\n'.join((
            f'N = {len(df_rex)}',
            f'Pearson $r$ = {r_rex:.3f}',
            f'RMSE = {rmse_rex:.3f} s⁻¹',
            f'MAE = {mae_rex:.3f} s⁻¹'))
        ax3.text(0.05, 0.95, stats_text_rex, transform=ax3.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    else:
        ax3.text(0.5, 0.5, "Insufficient overlapping\nR_ex data to plot.", 
                 ha='center', va='center', fontsize=14, color='gray')
        ax3.set_title('Chemical Exchange (R_ex)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_plot, format=file_format, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_plot}.{file_format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ground truth S2, Tau_e, and R_ex values generated by 'generate_example_data.py3' against SPINDLE predictions.")
    
    parser.add_argument("-t", "--truth_data", required=True, help="Path to the ground truth CSV file")
    parser.add_argument("-p", "--predicted", required=True, help="Path to the ML predicted CSV file")
    parser.add_argument("-o", "--output", help="Name of the output plot image", default="dynamics_comparison_plot.png")
    parser.add_argument("-f", "--file_format", dest="file_format", choices=['pdf','png','svg'], default='pdf', help="Output file format (default: pdf)")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    
    if not os.path.exists(args.truth_data):
        print(f"Error: Could not find truth file '{args.truth_data}'")
    elif not os.path.exists(args.predicted):
        print(f"Error: Could not find predicted file '{args.predicted}'")
    else:
        compare_dynamics_data(args.truth_data, args.predicted, args.output, args.file_format)

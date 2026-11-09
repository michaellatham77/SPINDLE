#!/usr/bin/env python3
# analyze_and_correct_2field.py3
import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description="Analyze 2-field ensemble predictions and apply post-hoc corrections")
    parser.add_argument("-results_file", type=str, required=True, help="Path to the input results npz file (e.g., final_exam_results_2field.npz)")
    parser.add_argument("-Rex_vs_size", action="store_true", help="Flag to see if there is correlation with SPINDLE prediction of Rex vs protein size")
    parser.add_argument("-error_analysis", action="store_true", help="Flag to analyze prediction errors")
    parser.add_argument("-correct_errors", action="store_true", help="Flag to turn on error calibration")
    parser.add_argument("-plot_file", type=str, default="final_performance_plots_2field.png", help="Path to save the final 2-field performance plots")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Load Results
    print(f"Loading 2-field results from {args.results_file}...")
    results = np.load(args.results_file, allow_pickle=True)
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']

    # Unpack all data into flat lists for analysis
    true_tauC, pred_tauC, pred_tauC_err = [], [], []
    if args.Rex_vs_size:
        true_tauC_per_res = []
    true_S2, pred_S2, pred_S2_err = [], [], []
    true_tauE, pred_tauE, pred_tauE_err = [], [], []
    true_Rex, pred_Rex, pred_Rex_err = [], [], []

    for true, pred in zip(true_labels, predicted_labels):
        true_tauC.append(true['tauC_ns'])
        pred_tauC.append(pred['tauC_ns'])
        pred_tauC_err.append(pred['tauC_err'])

        if args.Rex_vs_size:
            n_res = len(true['Rex'])
            true_tauC_per_res.extend([true['tauC_ns']] * n_res)
        
        true_S2.extend(true['S2'])
        pred_S2.extend(pred['S2'])
        pred_S2_err.extend(pred['S2_err'])
        
        true_tauE.extend(true['tauE_ns'] * 1000.0)
        pred_tauE.extend(pred['tauE_ns'] * 1000.0)
        pred_tauE_err.extend(pred['tauE_err_ns'] * 1000.0)
        
        true_Rex.extend(true['Rex'])
        pred_Rex.extend(pred['Rex'])
        pred_Rex_err.extend(pred['Rex_err'])

    # Convert lists to numpy arrays so we can do math on them
    true_tauC = np.array(true_tauC)
    pred_tauC = np.array(pred_tauC)
    pred_tauC_err = np.array(pred_tauC_err)
    if args.Rex_vs_size:
        true_tauC_per_res = np.array(true_tauC_per_res)

    true_S2 = np.array(true_S2)
    pred_S2 = np.array(pred_S2)
    pred_S2_err = np.array(pred_S2_err)

    true_tauE = np.array(true_tauE)
    pred_tauE = np.array(pred_tauE)
    pred_tauE_err = np.array(pred_tauE_err)

    true_Rex = np.array(true_Rex)
    pred_Rex = np.array(pred_Rex)
    pred_Rex_err = np.array(pred_Rex_err)

    # Post-Hoc Correction for tauC 
    print("\n--- Deriving Post-Hoc Correction for tauC ---")
    # Reshape for scikit-learn
    X = pred_tauC.reshape(-1, 1)
    y = true_tauC

    # Fit a linear regression model
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    print(f"Correction formula: Corrected_tauC = {slope:.4f} * Predicted_tauC + {intercept:.4f}")

    # Apply the correction
    corrected_pred_tauC = slope * pred_tauC + intercept
    corrected_tauC_err = pred_tauC_err * np.abs(slope)

    # Calculate MAE before and after correction
    mae_before = mean_absolute_error(true_tauC, pred_tauC)
    mae_after = mean_absolute_error(true_tauC, corrected_pred_tauC)
    print(f"MAE for tauC before correction: {mae_before:.3f} ns")
    print(f"MAE for tauC after correction:  {mae_after:.3f} ns")

    # Post-Hoc Corrections for tauE, S2, and Rex
    print("\n--- Deriving Calibration Corrections ---")

    # 1. Correct S2 (Linear Fit)
    corrected_pred_S2 = pred_S2
    # Clamp to physical limits [0, 1]
    corrected_pred_S2 = np.clip(corrected_pred_S2, 0.0, 1.0)
    corrected_S2_err = pred_S2_err

    # 2. Correct tauE (Linear Fit)
    corrected_pred_tauE = pred_tauE
    # Clamp to physical limits (non-negative)
    corrected_pred_tauE = np.maximum(corrected_pred_tauE, 0)
    corrected_tauE_err = pred_tauE_err

    # 3. Correct Rex (Quadratic Fit)
    corrected_pred_Rex = pred_Rex
    # Clamp to physical limits (non-negative)
    corrected_pred_Rex = np.maximum(corrected_pred_Rex, 0)
    corrected_Rex_err = pred_Rex_err

    # Metrics on corrected data
    def print_stats(name, y_true, y_pred, unit=""):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r, _ = pearsonr(y_true, y_pred)
        print(f"{name:10s} | MAE: {mae:.4f} {unit} | RMSE: {rmse:.4f} {unit} | Pearson r: {r:.4f}")

    print("\n" + "="*60)
    print("FINAL STATISTICAL REPORT (on 2-Field Corrected Data)")
    print("="*60)

    # 1. Standard Regression Metrics
    print_stats("Global tauC", true_tauC, corrected_pred_tauC, "ns")
    print_stats("S2", true_S2, corrected_pred_S2, "")
    print_stats("tauE", true_tauE, corrected_pred_tauE, "ps")
    print_stats("Rex", true_Rex, corrected_pred_Rex, "s^-1")

    # 2. Rex Specific: Classification Metrics
    THRESHOLD = 1.0 
    binary_true = (true_Rex > THRESHOLD).astype(int)
    binary_pred = (corrected_pred_Rex > THRESHOLD).astype(int)

    print("-" * 60)
    print("Rex CLASSIFICATION REPORT (Threshold > 1.0 s^-1)")
    print("-" * 60)

    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = recall_score(binary_true, binary_pred) 
    precision = precision_score(binary_true, binary_pred) 
    f1 = f1_score(binary_true, binary_pred) 

    print(f"Total Residues:    {len(true_Rex)}")
    print(f"True Exchange:     {np.sum(binary_true)} residues")
    print(f"Detected Exchange: {np.sum(binary_pred)} residues\n")
    print(f"Sensitivity (Recall): {sensitivity:.2%}  (Did we find the exchange?)")
    print(f"Precision:            {precision:.2%}  (Did we avoid false alarms?)")
    print(f"F1-Score:             {f1:.4f}      (Overall classification quality)")
    print("-" * 60)

    # 3. Rex Specific: Quantitative Accuracy on Active Residues Only
    active_indices = np.where(true_Rex > THRESHOLD)
    if len(active_indices[0]) > 0:
        active_true = np.array(true_Rex)[active_indices]
        active_pred = np.array(corrected_pred_Rex)[active_indices]
        
        mae_active = mean_absolute_error(active_true, active_pred)
        r_active, _ = pearsonr(active_true, active_pred)
        
        print(f"Rex Quantitative Accuracy (Active Residues Only):")
        print(f"MAE: {mae_active:.4f} s^-1 | Pearson r: {r_active:.4f}")
    print("="*60)

    if args.Rex_vs_size:
        # ---------------------------------------------------------
        # Rex CLASSIFICATION STRATIFIED BY PROTEIN SIZE (tauC)
        # ---------------------------------------------------------
        print("\n" + "-" * 60)
        print("Rex CLASSIFICATION BY PROTEIN SIZE (Testing the R2 Baseline Hypothesis)")
        print("-" * 60)

        tauC_bins = [
            ("Small (< 10 ns)", 0, 10),
            ("Medium (10 - 20 ns)", 10, 20),
            ("Large (> 20 ns)", 20, 50)
        ]

        for label, low, high in tauC_bins:
            mask = (true_tauC_per_res >= low) & (true_tauC_per_res < high)

            if np.sum(mask) == 0:
                continue

            bin_true_Rex = true_Rex[mask]
            bin_pred_Rex = corrected_pred_Rex[mask]

            bin_binary_true = (bin_true_Rex > THRESHOLD).astype(int)
            bin_binary_pred = (bin_pred_Rex > THRESHOLD).astype(int)

            sens = recall_score(bin_binary_true, bin_binary_pred, zero_division=0)
            prec = precision_score(bin_binary_true, bin_binary_pred, zero_division=0)
            f1 = f1_score(bin_binary_true, bin_binary_pred, zero_division=0)

            print(f"{label:20s} | Residues: {np.sum(mask):<8d} | Sens: {sens:.2%} | Prec: {prec:.2%} | F1: {f1:.4f}")

    if args.error_analysis:
        # ---------------------------------------------------------
        # EPISTEMIC UNCERTAINTY (COVERAGE PROBABILITY)
        # ---------------------------------------------------------
        print("\n" + "-" * 60)
        print("ENSEMBLE UNCERTAINTY VALIDATION (95% Confidence Interval)")
        print("-" * 60)

        def calc_coverage(true_val, pred_val, err_val, name):
            lower_bound = pred_val - (1.96 * err_val)
            upper_bound = pred_val + (1.96 * err_val)
            in_bounds = (true_val >= lower_bound) & (true_val <= upper_bound)
            coverage = np.mean(in_bounds) * 100.0
            print(f"{name:10s} | True value falls within 95% CI: {coverage:.2f}% of the time")
            return coverage

        calc_coverage(true_tauC, corrected_pred_tauC, corrected_tauC_err, "tauC")
        calc_coverage(true_S2, corrected_pred_S2, corrected_S2_err, "S2")
        calc_coverage(true_tauE, corrected_pred_tauE, corrected_tauE_err, "tauE")

        active_indices = np.where(true_Rex > THRESHOLD)
        if len(active_indices[0]) > 0:
            calc_coverage(np.array(true_Rex)[active_indices],
                          np.array(corrected_pred_Rex)[active_indices],
                          np.array(corrected_Rex_err)[active_indices],
                          "Rex (Active)")

        # ---------------------------------------------------------
        # ENSEMBLE ERROR VS. ACTUAL ERROR CORRELATION
        # ---------------------------------------------------------
        print("\n" + "-" * 60)
        print("ERROR CORRELATION (Does the ensemble know when it's wrong?)")
        print("-" * 60)

        def calc_error_correlation(true_val, pred_val, err_val, name):
            actual_error = np.abs(true_val - pred_val)
            corr, p_value = pearsonr(actual_error, err_val)
            print(f"{name:12s} | Pearson r: {corr:.4f}")
            return corr

        calc_error_correlation(true_tauC, corrected_pred_tauC, corrected_tauC_err, "tauC")
        calc_error_correlation(true_S2, corrected_pred_S2, corrected_S2_err, "S2")
        calc_error_correlation(true_tauE, corrected_pred_tauE, corrected_tauE_err, "tauE")
        calc_error_correlation(true_Rex, corrected_pred_Rex, corrected_Rex_err, "Rex (All)")

        if len(active_indices[0]) > 0:
            calc_error_correlation(np.array(true_Rex)[active_indices],
                                   np.array(corrected_pred_Rex)[active_indices],
                                   np.array(corrected_Rex_err)[active_indices],
                                   "Rex (Active)")

    if args.correct_errors:
        # ---------------------------------------------------------
        # CALCULATE CALIBRATION MULTIPLIERS FOR 95% COVERAGE
        # ---------------------------------------------------------
        print("\n" + "-" * 60)
        print("CALCULATING ERROR SCALING MULTIPLIERS (To hit exactly 95% CI)")
        print("-" * 60)

        def calculate_multiplier(true_val, pred_val, err_val, name, target_coverage=0.95):
            safe_err = np.where(err_val == 0, 1e-8, err_val)
            ratios = np.abs(true_val - pred_val) / (1.96 * safe_err)
            multiplier = np.percentile(ratios, target_coverage * 100)
            print(f"{name:12s} | Multiply raw ensemble errors by: {multiplier:.3f}")
            return multiplier

        def calibrate_ambiguity_threshold(true_vals, corrected_preds, corrected_errors, param_name, error_ceiling):
            abs_errors = np.abs(corrected_preds - true_vals)
            is_truly_ambiguous = (abs_errors > error_ceiling).astype(int)

            if np.sum(is_truly_ambiguous) == 0:
                print(f"{param_name:12s} | No errors exceeded the ceiling of {error_ceiling}. Cannot optimize threshold.")
                return 0.0

            precisions, recalls, thresholds = precision_recall_curve(is_truly_ambiguous, corrected_errors)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)

            if best_idx >= len(thresholds):
                best_idx = len(thresholds) - 1

            optimal_threshold = thresholds[best_idx]
            print(f"{param_name:12s} | Max F1: {f1_scores[best_idx]:.3f} | Prec: {precisions[best_idx]:.2%} | Rec: {recalls[best_idx]:.2%} | OPTIMAL THRESHOLD: {optimal_threshold:.4f}")
            return optimal_threshold

        tauC_mult = calculate_multiplier(true_tauC, corrected_pred_tauC, corrected_tauC_err, "tauC")
        S2_mult = calculate_multiplier(true_S2, corrected_pred_S2, corrected_S2_err, "S2")
        tauE_mult = calculate_multiplier(true_tauE, corrected_pred_tauE, corrected_tauE_err, "tauE")
        Rex_mult = calculate_multiplier(true_Rex, corrected_pred_Rex, corrected_Rex_err, "Rex")

        # ---------------------------------------------------------
        # NEW: CALCULATE OPTIMAL QUALITY CONTROL AMBIGUITY THRESHOLDS
        # ---------------------------------------------------------
        print("\n" + "-" * 60)
        print("CALCULATING OPTIMAL AMBIGUITY THRESHOLDS (via F1-Optimization)")
        print("-" * 60)

        ceilings = {
            "S2": 0.10,       
            "tauE": 150.0,    
            "Rex": 2.5        
        }

        thresh_s2 = calibrate_ambiguity_threshold(true_S2, corrected_pred_S2, corrected_S2_err, "S2", ceilings["S2"])
        thresh_taue = calibrate_ambiguity_threshold(true_tauE, corrected_pred_tauE, corrected_tauE_err, "tauE", ceilings["tauE"])
        thresh_rex = calibrate_ambiguity_threshold(true_Rex, corrected_pred_Rex, corrected_Rex_err, "Rex", ceilings["Rex"])

    # --- Generate Final Plots ---
    print("\nGenerating final performance plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Final 2-Field Model Performance', fontsize=16)

    # Plot corrected tauC
    axes[0, 0].scatter(true_tauC, corrected_pred_tauC, alpha=0.1)
    axes[0, 0].plot([0, 35], [0, 35], 'r--')
    axes[0, 0].set_title('Global tau_c (Corrected)')
    axes[0, 0].set_xlabel('True (ns)')
    axes[0, 0].set_ylabel('Predicted (ns)')
    axes[0, 0].set_xlim(0, 35)
    axes[0, 0].set_ylim(0, 35)
    axes[0, 0].grid(True)

    # Plot S2
    axes[0, 1].scatter(true_S2, corrected_pred_S2, alpha=0.01)
    axes[0, 1].plot([0, 1], [0, 1], 'r--')
    axes[0, 1].set_title('S²')
    axes[0, 1].set_xlabel('True')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True)

    # Plot tauE
    axes[1, 0].scatter(true_tauE, corrected_pred_tauE, alpha=0.01)
    max_val_taue = max(np.max(pred_tauE), np.max(true_tauE))
    axes[1, 0].plot([0, max_val_taue], [0, max_val_taue], 'r--')
    axes[1, 0].set_title('tau_E')
    axes[1, 0].set_xlabel('True (ps)')
    axes[1, 0].set_ylabel('Predicted (ps)')
    axes[1, 0].grid(True)

    # Plot Rex
    axes[1, 1].scatter(true_Rex, corrected_pred_Rex, alpha=0.01)
    axes[1, 1].plot([0, 20], [0, 20], 'r--')
    axes[1, 1].set_title('R_ex (relative to 600 MHz)')
    axes[1, 1].set_xlabel('True')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_dir = os.path.dirname(os.path.abspath(args.plot_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(args.plot_file)
    print(f"Saved final plots to '{args.plot_file}'")

if __name__ == "__main__":
    main()

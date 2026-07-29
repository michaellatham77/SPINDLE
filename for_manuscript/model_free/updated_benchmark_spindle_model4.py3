#!/usr/bin/env python3


import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from spindle_calibrations import CALIBRATIONS # Import from our single source of truth

# --- CONFIGURATION ---
USE_CPU_ONLY = True 

# --- PARSER ---
parser = argparse.ArgumentParser(description="Get bulk modelfree run results and compare to ground truth (Model 4)")
parser.add_argument('-model', type=str, default='./model', help='Top level model directory')
parser.add_argument('-field', type=float, default=800.0, help='Proton resonance frequency in MHz (e.g., 800.0)')
parser.add_argument('-data_file', type=str, default='modelfree_model4_dataset.npz', help='Compressed numpy ground truth file')
parser.add_argument('--plot', action='store_true', help='Generate correlation plot of True vs Predicted values')
parser.add_argument('-plot_file', type=str, default='tf_model4_predictions_correlation.pdf', help='Output high-def PDF for the correlation scatter plots')

if len(sys.argv) == 1 and not sys.stdin.isatty():
    pass

args = parser.parse_args()

# Dynamically construct the model directory path
MODEL_DIR = os.path.join(args.model, str(int(args.field)))
DATA_FILE = args.data_file
N_MODELS = 10

# --- VERIFY CALIBRATION EXISTS ---
if args.field not in CALIBRATIONS:
    print(f"Error: No calibration data found for field {args.field} MHz in spindle_config.py")
    print(f"Available fields: {list(CALIBRATIONS.keys())}")
    sys.exit(1)

cal = CALIBRATIONS[args.field]

# --- 1. GPU/CPU Setup ---
if USE_CPU_ONLY:
    print("[Config] Forcing CPU execution...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print("[Config] Attempting to use GPU...")

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# --- PLOTTING FUNCTION ---
def plot_results(y_true_s2, y_pred_s2, y_true_te, y_pred_te, y_true_rex, y_pred_rex, y_true_tc, y_pred_tc, outfile):
    """Generates scatter plots comparing True vs Predicted data for Model 4."""
    print(f"Generating correlation plot: {outfile}")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # S2 Plot
    if len(y_true_s2) > 0:
        axes[0].scatter(y_true_s2, y_pred_s2, alpha=0.5, s=10, color='blue')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5) # y=x reference line
        axes[0].set_title('Order Parameter ($S^2$)')
        axes[0].set_xlabel('True $S^2$')
        axes[0].set_ylabel('Predicted $S^2$')
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

    # tauE Plot
    if len(y_true_te) > 0:
        max_te = max(y_true_te.max(), y_pred_te.max())
        axes[1].scatter(y_true_te, y_pred_te, alpha=0.5, s=10, color='green')
        axes[1].plot([0, max_te], [0, max_te], 'k--', linewidth=1.5)
        axes[1].set_title(r'Internal Correlation Time ($\tau_e$, ns)')
        axes[1].set_xlabel(r'True $\tau_e$')
        axes[1].set_ylabel(r'Predicted $\tau_e$')

    # Rex Plot
    if len(y_true_rex) > 0:
        max_rex = max(y_true_rex.max(), y_pred_rex.max())
        axes[2].scatter(y_true_rex, y_pred_rex, alpha=0.5, s=10, color='purple')
        axes[2].plot([0, max_rex], [0, max_rex], 'k--', linewidth=1.5)
        axes[2].set_title(r'Conformational Exchange ($R_{ex}$, s$^{-1}$)')
        axes[2].set_xlabel(r'True $R_{ex}$')
        axes[2].set_ylabel(r'Predicted $R_{ex}$')

    # tauC Plot
    if len(y_true_tc) > 0:
        max_tc = max(y_true_tc.max(), y_pred_tc.max())
        axes[3].scatter(y_true_tc, y_pred_tc, alpha=0.6, s=30, color='red')
        axes[3].plot([0, max_tc], [0, max_tc], 'k--', linewidth=1.5)
        axes[3].set_title(r'Global Tumbling Time ($\tau_c$, ns)')
        axes[3].set_xlabel(r'True $\tau_c$')
        axes[3].set_ylabel(r'Predicted $\tau_c$')

    plt.tight_layout()
    # Explicitly save as high definition vector PDF
    plt.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# --- 2. Load Data ---
print(f"Loading data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found.")
    sys.exit(1)

data = np.load(DATA_FILE, allow_pickle=True)
features_list = data['features']
labels_list = data['labels']

print(f"Processing {len(features_list)} proteins...")

# --- 3. Load Models ---
print(f"Loading {N_MODELS} models from {MODEL_DIR}...")
ensemble_models = []
for i in range(N_MODELS):
    p1 = os.path.join(MODEL_DIR, f'model_{i}.keras')
    p2 = os.path.join(MODEL_DIR, f'model_{i}.h5')
    path = p1 if os.path.exists(p1) else p2
    
    if os.path.exists(path): 
        ensemble_models.append(tf.keras.models.load_model(path, compile=False))

if not ensemble_models:
    print("Error: No models loaded. Check your --model and --field paths.")
    sys.exit(1)

# --- 4. Run Predictions & Corrections ---
print(f"Generating predictions using {args.field} MHz calibrations...")

true_s2, true_te, true_rex, true_tc = [], [], [], []
pred_s2, pred_te, pred_rex, pred_tc = [], [], [], []

for features, record in tqdm(zip(features_list, labels_list), total=len(features_list)):
    
    # Ground Truth
    n_res = len(record['S2'])
    true_s2.append(record['S2'])
    true_te.append(record['tauE_ns'])
    true_rex.append(record['Rex'])
    
    tc_val = record['tauC_ns']
    tc_val = tc_val[0] if isinstance(tc_val, (list, np.ndarray)) else tc_val
    true_tc.append(np.full(n_res, tc_val))

    # Prediction
    features_batch = np.expand_dims(features, axis=0).astype(np.float32)
    
    model_preds_local = []
    model_preds_global = []
    
    for model in ensemble_models:
        local_p, global_p = model.predict(features_batch, verbose=0)
        model_preds_local.append(local_p.squeeze())
        model_preds_global.append(global_p.squeeze())
    
    avg_local = np.mean(model_preds_local, axis=0) # Shape (N, 3)
    avg_global = np.mean(model_preds_global)       # Scalar
    
    # --- RAW PREDICTIONS ---
    raw_te_ns = avg_local[:, 0]
    raw_s2    = avg_local[:, 1]
    raw_rex   = avg_local[:, 2]
    raw_tc_ns = avg_global
    
    # --- DYNAMIC CORRECTIONS ---
    
    # 1. tauC
    corr_tc_ns = (cal['TAUC']['slope'] * raw_tc_ns) + cal['TAUC']['intercept']
    
    # 2. S2
    corr_s2 = (cal['S2']['slope'] * raw_s2) + cal['S2']['intercept']
    
    # 3. tauE (ns -> ps -> Correct -> ns)
    raw_te_ps = raw_te_ns * 1000.0
    corr_te_ps = (cal['TAUE']['slope'] * raw_te_ps) + cal['TAUE']['intercept']
    corr_te_ns = corr_te_ps / 1000.0
    
    # 4. Rex (Quadratic Correction using A, B, C parameters)
    corr_rex = (cal['REX']['a'] * (raw_rex**2)) + (cal['REX']['b'] * raw_rex) + cal['REX']['c']
    
    pred_s2.append(corr_s2)
    pred_te.append(corr_te_ns)
    pred_rex.append(corr_rex)
    pred_tc.append(np.full(n_res, corr_tc_ns))

# --- 5. Final Evaluation ---
y_true_s2 = np.concatenate(true_s2)
y_true_te = np.concatenate(true_te)
y_true_rex = np.concatenate(true_rex)
y_true_tc = np.concatenate(true_tc)

y_pred_s2 = np.concatenate(pred_s2)
y_pred_te = np.concatenate(pred_te)
y_pred_rex = np.concatenate(pred_rex)
y_pred_tc = np.concatenate(pred_tc)

rmse_s2 = np.sqrt(mean_squared_error(y_true_s2, y_pred_s2))
rmse_te = np.sqrt(mean_squared_error(y_true_te, y_pred_te))
rmse_rex = np.sqrt(mean_squared_error(y_true_rex, y_pred_rex))
rmse_tc = np.sqrt(mean_squared_error(y_true_tc, y_pred_tc))

table_data = {
    'True_S2': y_true_s2,
    'Pred_S2': y_pred_s2,
    'True_tauE_ns': y_true_te,
    'Pred_tauE_ns': y_pred_te,
    'True_Rex': y_true_rex,
    'Pred_Rex': y_pred_rex,
    'True_tauC_ns': y_true_tc,
    'Pred_tauC_ns': y_pred_tc
}

# Create a Pandas DataFrame (Table)
results_df = pd.DataFrame(table_data)

# Print the first and last few rows to the terminal
print("\n--- Model Predictions vs Ground Truth ---")
print(results_df.head(20)) # Prints the first 20 rows
print("...")

# Save the full table to a CSV file so you can inspect all of it
csv_filename = f"model4_predictions_{int(args.field)}MHz.csv"
results_df.to_csv(csv_filename, index=False)
print(f"\nFull table saved successfully to: {csv_filename}")


print("\n" + "="*50)
print(f"   TENSORFLOW RESULTS MODEL 4 ({args.field} MHz)")
print("="*50)
print(f"{'Parameter':<10} | {'RMSE':<15}")
print("-" * 50)
print(f"{'S2':<10} | {rmse_s2:<15.5f}")
print(f"{'tauE':<10} | {rmse_te:<15.5f} ns")
print(f"{'Rex':<10} | {rmse_rex:<15.5f} s^-1")
print(f"{'tauC':<10} | {rmse_tc:<15.5f} ns")
print("="*50)

# Generate Plot if requested
if args.plot:
    plot_results(y_true_s2, y_pred_s2, y_true_te, y_pred_te, y_true_rex, y_pred_rex, y_true_tc, y_pred_tc, args.plot_file)

#!/usr/bin/env python3


import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, confusion_matrix

# --- 1. SETUP ---
parser = argparse.ArgumentParser()
parser.add_argument('-mf_dir', type=str, required=True)
parser.add_argument('-data_file', type=str, required=True)
parser.add_argument('-model_path', type=str, default='./model')
parser.add_argument('-field', type=float, default=800.0)
parser.add_argument('-out_prefix', type=str, default='benchmark')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

try:
    from spindle_calibrations import CALIBRATIONS
    cal = CALIBRATIONS[args.field]
except:
    print("Error: spindle_calibrations.py not found."); sys.exit(1)

# --- 2. PARSERS ---

def parse_mfout_exact(file_path):
    """Parses parameter values from mfout.final."""
    results = {}
    tm_val = np.nan
    if not os.path.exists(file_path): return results, tm_val
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_param = None
    for line in lines:
        parts = line.split()
        if not parts: continue
        
        if parts[0] == 'tm' and len(parts) > 2 and parts[1] == '(ns)':
            try: tm_val = float(parts[2])
            except: pass
            
        if parts[0] == 'S2' and '()' in parts: current_param = 'S2'; continue
        elif parts[0] == 'te' and '(ps)' in parts: current_param = 'te'; continue
        elif parts[0] == 'Rex' and '(1/s)' in parts: current_param = 'Rex'; continue
        elif parts[0] == 'S2f' and '()' in parts: current_param = 'S2f'; continue
        elif parts[0] == 'S2s' and '()' in parts: current_param = 'S2s'; continue
        elif parts[0] in ['stop_', 'R1', 'R2', 'NOE', 'Theta']: current_param = None; continue
            
        if current_param and parts[0].isdigit() and len(parts) >= 2:
            res = int(parts[0])
            try:
                val = float(parts[1]) if '*' not in parts[1] else np.nan
                if res not in results: results[res] = {}
                results[res][current_param] = val
            except: pass

    # Apply mathematical inference as a fallback ONLY
    for res in results:
        s2 = results[res].get('S2', np.nan)
        te = results[res].get('te', np.nan)
        rex = results[res].get('Rex', np.nan)
        s2f = results[res].get('S2f', np.nan)
        s2s = results[res].get('S2s', np.nan)

        if not pd.isna(s2f) and not pd.isna(s2s) and s2s > 1e-4:
            results[res]['S2'] = s2f * s2s

        has_te = not pd.isna(te) and te > 1e-4
        has_rex = not pd.isna(rex) and rex > 1e-4
        has_s2s = not pd.isna(s2s) and s2s > 1e-4 and abs(s2s - 1.0) > 1e-4 and abs(s2s - s2) > 1e-4

        if has_s2s: results[res]['Mod'] = 5
        elif has_te and has_rex: results[res]['Mod'] = 4
        elif has_rex: results[res]['Mod'] = 3
        elif has_te: results[res]['Mod'] = 2
        elif not pd.isna(s2) and s2 > 1e-4: results[res]['Mod'] = 1
        else: results[res]['Mod'] = np.nan

    return results, tm_val

def parse_model_from_table(file_path):
    """Extracts explicit models from ANY .par file, handling blank columns perfectly."""
    models = {}
    excluded = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # A valid data row starts with a residue number
                if len(parts) >= 1 and parts[0].isdigit():
                    res = int(parts[0])
                    # If the 2nd column is 1-5, it's a valid model
                    if len(parts) > 1 and parts[1] in ['1', '2', '3', '4', '5']:
                        models[res] = int(parts[1])
                    else:
                        # Otherwise, it's an excluded residue
                        excluded.add(res)
    except Exception: pass
    return models, excluded

# --- 3. SCAN & PROCESS ---
print(f"Deep scanning {args.mf_dir}...")
found_data = {}
for root, dirs, files in os.walk(args.mf_dir):
    if "mfout.final" in files:
        match = re.search(r'protein_(\d+)', root)
        if match:
            p_num = int(match.group(1))
            res_data, tm = parse_mfout_exact(os.path.join(root, "mfout.final"))
            
            # --- THE FILE SELECTION FIX ---
            par_files = [f for f in files if f.endswith('.par')]
            highest_par = None
            max_score = -1
            
            for pf in par_files:
                # 1. Specifically extract the number right after iter or inter
                m_iter = re.search(r'(?:iter|inter)(\d+)', pf, re.IGNORECASE)
                if m_iter:
                    it = int(m_iter.group(1))
                    is_iter = True
                else:
                    # Fallback: grab the last number before .par
                    num_match = re.search(r'(\d+)\.par$', pf, re.IGNORECASE)
                    it = int(num_match.group(1)) if num_match else 0
                    is_iter = False
                    
                # 2. Add 100,000 to the score if it contains 'iter' or 'inter'
                # This guarantees protein_00620.inter2.par beats protein_00620.par
                score = it + (100000 if is_iter else 0)
                
                if score > max_score:
                    max_score = score
                    highest_par = pf
                        
            # If we extracted models from the highest .par file, completely override the fallback
            if highest_par:
                best_par_models, best_par_excluded = parse_model_from_table(os.path.join(root, highest_par))
                if best_par_models:
                    for r in list(res_data.keys()):
                        if r in best_par_models:
                            res_data[r]['Mod'] = best_par_models[r]
                        else:
                            # If it wasn't assigned 1-5 in the .par file, strictly exclude it
                            res_data[r]['Mod'] = np.nan
                            res_data[r]['S2'] = np.nan
                            res_data[r]['te'] = np.nan
                            res_data[r]['Rex'] = np.nan
                        
            found_data[p_num] = {'res': res_data, 'tm': tm}

print(f"Processing {len(found_data)} proteins...")
data = np.load(args.data_file, allow_pickle=True)
features_list, labels_list = data['features'], data['labels']

m_dir = os.path.join(args.model_path, str(int(args.field)))
ensemble = [tf.keras.models.load_model(os.path.join(m_dir, f'model_{i}.keras')) 
            for i in range(10) if os.path.exists(os.path.join(m_dir, f'model_{i}.keras'))]

all_results = []
for idx, (features, record) in tqdm(enumerate(zip(features_list, labels_list)), total=len(features_list)):
    p_id = idx + 1
    mf_match = found_data.get(p_id, {})
    mf_res = mf_match.get('res', {})
    mf_tc = mf_match.get('tm', np.nan)

    # AI Preds
    ai_s2, ai_te, ai_rex, ai_tc = None, None, None, np.nan
    if ensemble:
        feat_batch = np.expand_dims(features, axis=0).astype(np.float32)
        preds = [m.predict(feat_batch, verbose=0) for m in ensemble]
        avg_l = np.mean([p[0].squeeze() for p in preds], axis=0)
        avg_g = np.mean([p[1].squeeze() for p in preds])
        ai_s2 = (cal['S2']['slope'] * avg_l[:, 1]) + cal['S2']['intercept']
        ai_te = ((cal['TAUE']['slope'] * (avg_l[:, 0] * 1000.0)) + cal['TAUE']['intercept']) / 1000.0
        ai_rex = (cal['REX']['a'] * (avg_l[:, 2]**2)) + (cal['REX']['b'] * avg_l[:, 2]) + cal['REX']['c']
        ai_tc = (cal['TAUC']['slope'] * avg_g) + cal['TAUC']['intercept']

    t_s2 = record['S2']
    t_tc = record['tauC_ns'][0] if isinstance(record['tauC_ns'], (list, np.ndarray)) else record['tauC_ns']
    
    # Ground Truth Model Processing
    if 'model' in record:
        t_mod = record['model']
        # If the GT array is 0-indexed (0,1,2,3,4), shift it to 1-5
        if len(t_mod) > 0 and np.min(t_mod) == 0 and np.max(t_mod) <= 4:
            t_mod = [int(m) + 1 for m in t_mod]
        else:
            t_mod = [int(m) for m in t_mod]
    else:
        t_mod = []
        for i in range(len(t_s2)):
            te_val = record['tauE_ns'][i]
            rex_val = record['Rex'][i]
            if te_val > 1e-5 and rex_val > 1e-5: t_mod.append(4)
            elif rex_val > 1e-5: t_mod.append(3)
            elif te_val > 1e-5: t_mod.append(2)
            else: t_mod.append(1)

    for i in range(len(t_s2)):
        res_key = i + 1
        res_data = mf_res.get(res_key, {})
        mf_te_val = res_data.get('te', np.nan)
        
        all_results.append({
            "Protein": p_id,
            "Residue": res_key,
            "True_S2": t_s2[i], "True_te": record['tauE_ns'][i], "True_Rex": record['Rex'][i], 
            "True_TC": t_tc, "True_Model": t_mod[i],
            "MF_S2": res_data.get('S2', np.nan),
            "MF_te": mf_te_val / 1000.0 if not pd.isna(mf_te_val) else np.nan,
            "MF_Rex": res_data.get('Rex', np.nan),
            "MF_TC": mf_tc,
            "MF_Model": res_data.get('Mod', np.nan),
            "AI_S2": ai_s2[i] if ai_s2 is not None else np.nan,
            "AI_te": ai_te[i] if ai_te is not None else np.nan,
            "AI_Rex": ai_rex[i] if ai_rex is not None else np.nan,
            "AI_TC": ai_tc
        })

df = pd.DataFrame(all_results)
df.to_csv(f"{args.out_prefix}_data.csv", index=False)

# --- 4. OUTPUT TABLES & VERIFICATION ---
print("\n" + "="*60)
print(f"{'Parameter':15} | {'ModelFree RMSE':15} | {'AI RMSE':15}")
print("-" * 60)
for k, label in [('S2', 'S2'), ('te', 'tau_e'), ('Rex', 'Rex'), ('TC', 'tau_c')]:
    m_v = df.dropna(subset=[f'True_{k}', f'MF_{k}'])
    a_v = df.dropna(subset=[f'True_{k}', f'AI_{k}'])
    m_rmse = np.sqrt(mean_squared_error(m_v[f'True_{k}'], m_v[f'MF_{k}'])) if not m_v.empty else np.nan
    a_rmse = np.sqrt(mean_squared_error(a_v[f'True_{k}'], a_v[f'AI_{k}'])) if not a_v.empty else np.nan
    print(f"{label:15} | {m_rmse:15.4f} | {a_rmse:15.4f}")
print("="*60 + "\n")

# ---- PRINT MODEL VERIFICATION TO TERMINAL ----
print("="*60)
print("MODEL SELECTION VERIFICATION")
print("="*60)

gt_counts = df['True_Model'].value_counts().sort_index().to_dict()
mf_counts = df['MF_Model'].value_counts().sort_index().to_dict()

print(f"{'Model':<10} | {'Ground Truth Count':<20} | {'ModelFree Count':<20}")
print("-" * 60)
for m in [1, 2, 3, 4, 5]:
    print(f"Model {m:<4} | {gt_counts.get(m, 0):<20} | {mf_counts.get(m, 0):<20}")

print("\nSample of Model Alignments (First 25 Valid Residues):")
sample_df = df.dropna(subset=['MF_Model']).head(25)
# Ensure models are printed as integers rather than floats for clarity
sample_df['True_Model'] = sample_df['True_Model'].astype(int)
sample_df['MF_Model'] = sample_df['MF_Model'].astype(int)
print(sample_df[['Protein', 'Residue', 'True_Model', 'MF_Model']].to_string(index=False))
print("="*60 + "\n")


# --- 5. VISUALIZATION (SEPARATE PDFs) ---
print("Generating separate visualization PDFs...")

# 5A. Parameter Scatters
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
confs = [('MF_S2', 'True_S2', '$S^2$', 'blue', [0, 1]),
         ('MF_te', 'True_te', r'$\tau_e$ (ns)', 'green', [0, 1]),
         ('MF_Rex', 'True_Rex', '$R_{ex}$ (s$^{-1}$)', 'red', [0, 15]),
         ('MF_TC', 'True_TC', r'$\tau_c$ (ns)', 'purple', None)]

for ax, (px, py, title, col, lim) in zip(axes, confs):
    v = df.dropna(subset=[px, py])
    if not v.empty:
        ax.scatter(v[px], v[py], alpha=0.2, s=8, color=col)
        m = max(v[px].max(), v[py].max()) if lim is None else lim[1]
        ax.plot([0, m], [0, m], 'k--', alpha=0.7)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Fast ModelFree', fontsize=12)
        ax.set_ylabel('Ground Truth', fontsize=12)
        if lim: ax.set_xlim(lim); ax.set_ylim(lim)
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_scatters.pdf")
plt.close()

# 5B. Fit vs Excluded Bar Chart
plt.figure(figsize=(6, 5))
fit_count = df['MF_S2'].notna().sum()
excl_count = df['MF_S2'].isna().sum()
plt.bar(['Fit', 'Excluded'], [fit_count, excl_count], color=['#4CAF50', '#F44336'], edgecolor='black')
plt.title('Residues: Fit vs Excluded', fontweight='bold')
plt.ylabel('Total Count')
for i, v in enumerate([fit_count, excl_count]):
    plt.text(i, v + (max(fit_count, excl_count)*0.02), str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_fit_vs_excluded.pdf")
plt.close()

# 5C. Model Selection Bar Chart
plt.figure(figsize=(7, 5))
valid_models = df['MF_Model'].dropna()
valid_models = valid_models[valid_models > 0]
model_counts = valid_models.value_counts().reindex([1,2,3,4,5], fill_value=0)
plt.bar(model_counts.index, model_counts.values, color='skyblue', edgecolor='black')
plt.title('Fast ModelFree Model Selection', fontweight='bold')
plt.xlabel('Model Number')
plt.ylabel('Count')
plt.xticks([1, 2, 3, 4, 5])
for i, v in zip(model_counts.index, model_counts.values):
    plt.text(i, v + (max(model_counts.values)*0.02), str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_model_selection.pdf")
plt.close()

# 5D. Heatmap (Confusion Matrix)
plt.figure(figsize=(7, 6))
v_mod = df.dropna(subset=['True_Model', 'MF_Model'])
v_mod = v_mod[(v_mod['True_Model'] > 0) & (v_mod['MF_Model'] > 0)]
if not v_mod.empty:
    cm = confusion_matrix(v_mod['True_Model'], v_mod['MF_Model'], labels=[1,2,3,4,5])
    im = plt.imshow(cm, cmap='Blues')
    plt.title("Model Selection: Ground Truth vs ModelFree", fontweight='bold')
    plt.xlabel("Fast ModelFree Assigned Model")
    plt.ylabel("Ground Truth Model")
    plt.xticks(np.arange(5), [1,2,3,4,5])
    plt.yticks(np.arange(5), [1,2,3,4,5])
    
    thresh = cm.max() / 2.
    for i in range(5):
        for j in range(5):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontweight='bold')
    plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_heatmap.pdf")
plt.close()

print("Done! All separate PDFs generated successfully.")

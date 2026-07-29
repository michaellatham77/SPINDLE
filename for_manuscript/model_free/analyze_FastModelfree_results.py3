#!/usr/bin/env python3


#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_ground_truth(npz_file):
    print(f"Loading ground truth from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)
    return data['labels']

def parse_fmf_output(protein_dir):
    log_file = os.path.join(protein_dir, "fastMF.log")
    protein_name = os.path.basename(protein_dir)
    main_log_file = os.path.join(protein_dir, f"{protein_name}.log")

    global_tm = 0.0
    if os.path.exists(main_log_file):
        with open(main_log_file, 'r') as f:
            for line in f:
                if "Tensor: tm" in line:
                    try:
                        global_tm = float(line.split()[2])
                    except (IndexError, ValueError): pass

    if not os.path.exists(log_file): return None

    spin_to_model = {}
    spin_to_local_tm = {}
    
    with open(log_file, 'r') as f:
        current_model = None
        for line in f:
            if line.startswith("Model ") and "spins:" in line:
                current_model = int(line.split(" ")[1])
            elif line.startswith("Unassigned spins:"):
                current_model = None
            elif current_model is not None and line.strip() and not line.startswith("Model"):
                spins = line.split()
                for s in spins:
                    if s.isdigit(): spin_to_model[int(s)] = current_model
            
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == "Tm" and parts[0].isdigit():
                spin_to_local_tm[int(parts[0])] = float(parts[2])

    parsed_data = {}
    for spin, model in spin_to_model.items():
        assigned_tc = spin_to_local_tm.get(spin, global_tm)
        parsed_data[spin] = {'Model': model, 'S2': 0.0, 'tauE': 0.0, 'Rex': 0.0, 'tauC': assigned_tc}

    for model in range(1, 6):
        mfout_file = os.path.join(protein_dir, f"mfout.{model}")
        if not os.path.exists(mfout_file): continue

        current_param = None
        temp_s2_f = {} 
        
        with open(mfout_file, 'r') as f:
            for line in f:
                if "S2 " in line and "(" in line: current_param = "S2"
                elif "S2s" in line and "(" in line: current_param = "S2s"
                elif "te " in line and "(" in line: current_param = "tauE"
                elif "Rex " in line and "(" in line: current_param = "Rex"
                elif "stop_" in line or "corr_" in line: current_param = None
                
                parts = line.split()
                if current_param and len(parts) >= 2 and parts[0].isdigit():
                    spin = int(parts[0])
                    if spin_to_model.get(spin) == model:
                        try:
                            val = float(parts[1])
                            if current_param == "S2":
                                if model == 5: temp_s2_f[spin] = val
                                else: parsed_data[spin]['S2'] = val
                            elif current_param == "S2s" and model == 5:
                                parsed_data[spin]['S2'] = temp_s2_f.get(spin, 1.0) * val
                            elif current_param == "tauE":
                                parsed_data[spin]['tauE'] = min(val / 1000.0, 1.0)
                            elif current_param == "Rex":
                                parsed_data[spin]['Rex'] = val
                        except ValueError: pass

    return {spin - 1: data for spin, data in parsed_data.items()}

def main(npz_file, fmf_base_dir, out_prefix):
    labels = load_ground_truth(npz_file)
    true_s2, pred_s2, true_taue, pred_taue, true_rex, pred_rex, true_tauc, pred_tauc = [], [], [], [], [], [], [], []
    raw_true_models, pred_models_list = [], []
    all_selected_models = []
    total_residues, assigned_residues = 0, 0

    print(f"Parsing FastModelFree outputs...")
    for idx, label_dict in enumerate(labels):
        protein_dir = os.path.join(fmf_base_dir, f"protein_{idx+1:05d}")
        fmf_data = parse_fmf_output(protein_dir) if os.path.exists(protein_dir) else {}
            
        gt_s2 = label_dict.get('S2', [])
        gt_taue = label_dict.get('tauE_ns', [])
        gt_rex = label_dict.get('Rex', [])
        gt_model_raw = label_dict.get('Model', label_dict.get('model', None))
        gt_tauc_raw = label_dict.get('tauC_ns', label_dict.get('tauC', None))
        
        for res_idx in range(len(gt_s2)):
            total_residues += 1
            
            if gt_model_raw is not None:
                tm = int(gt_model_raw[res_idx])
            else:
                te_val = gt_taue[res_idx] if len(gt_taue) > res_idx else 0.0
                rex_val = gt_rex[res_idx] if len(gt_rex) > res_idx else 0.0
                thresh = 1e-4
                if rex_val <= thresh and te_val <= thresh: tm = 1
                elif rex_val <= thresh and te_val > thresh: tm = 2
                elif rex_val > thresh and te_val <= thresh: tm = 3
                else: tm = 4

            if fmf_data and res_idx in fmf_data:
                assigned_residues += 1
                pm = fmf_data[res_idx]['Model']
                
                raw_true_models.append(tm)
                pred_models_list.append(pm)
                all_selected_models.append(pm)
                
                true_s2.append(gt_s2[res_idx])
                pred_s2.append(fmf_data[res_idx]['S2'])
                true_taue.append(gt_taue[res_idx])
                pred_taue.append(fmf_data[res_idx]['tauE']) 
                true_rex.append(gt_rex[res_idx])
                pred_rex.append(fmf_data[res_idx]['Rex'])
                
                if gt_tauc_raw is not None:
                    gt_t = np.atleast_1d(gt_tauc_raw)
                    tc_val = float(gt_t[0]) if gt_t.size == 1 else float(gt_t[res_idx])
                    true_tauc.append(tc_val)
                    pred_tauc.append(fmf_data[res_idx]['tauC'])

    shift = 1 if len(raw_true_models) > 0 and np.min(raw_true_models) == 0 else 0
    true_models_list = [m + shift for m in raw_true_models]

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or '.', exist_ok=True)

    # 1. Scatters
    #fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    #confs = [((true_tauc, pred_tauc), r'$\tau_c$ (ns)', 'purple', None),
    #         ((true_s2, pred_s2), '$S^2$', 'blue', [0, 1]),
    #         ((true_taue, pred_taue), r'$\tau_e$ (ns)', 'green', [0, 1]),
    #         ((true_rex, pred_rex), '$R_{ex}$ (s$^{-1}$)', 'red', [0, 14])]
    #for ax, ((t, p), title, color, lim) in zip(axes, confs):
    #    ax.scatter(t, p, alpha=0.3, s=10, color=color)
    #    m = max(max(t), max(p)) if lim is None else lim[1]
    #    ax.plot([0, m], [0, m], 'k--', lw=2)
    #    ax.set_title(title); ax.set_xlim(lim); ax.set_ylim(lim)
    #plt.tight_layout(); plt.savefig(f"{out_prefix}_scatter.pdf")

    # 2. Distribution Bar Graph (Added numbers here)
    plt.figure(figsize=(8, 6))
    mc = Counter(all_selected_models)
    models = [1, 2, 3, 4, 5]
    counts = [mc.get(m, 0) for m in models]
    bars = plt.bar(models, counts, color='skyblue', edgecolor='black')
    plt.title('FastModelFree: Distribution of Selected Models')
    plt.xlabel('Model Number')
    plt.ylabel('Number of Residues')
    plt.xticks(models)
    
    # Text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(counts) * 0.01),
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig(f"{out_prefix}_bargraph.pdf")

    # 3. Model Comparison Heatmap (Intact logic from your script)
    print("Generating Model Comparison Matrix...")
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(true_models_list, pred_models_list):
        if 1 <= t <= 5 and 1 <= p <= 5: cm[t-1, p-1] += 1
    
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax).set_label('Number of Residues', rotation=270, labelpad=15)
    
    for (i, j), val in np.ndenumerate(cm):
        color = 'white' if val > (cm.max()/2) else 'black'
        ax.text(j, i, str(val), va='center', ha='center', color=color, fontweight='bold')
            
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([1, 2, 3, 4, 5]); ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel('Predicted Model (FastModelFree)'); ax.set_ylabel('True Model (Ground Truth)')
    ax.set_title('Model Assignment Accuracy', pad=20)
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(f"{out_prefix}_model_comparison.pdf", bbox_inches='tight')
    print(f"Success! All plots saved with prefix: {out_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-npz", required=True)
    parser.add_argument("-fmf_dir", required=True)
    parser.add_argument("-out_prefix", default="fmf_results")
    main(parser.parse_args().npz, parser.parse_args().fmf_dir, parser.parse_args().out_prefix)

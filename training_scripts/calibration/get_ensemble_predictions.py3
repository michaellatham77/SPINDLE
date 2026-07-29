#!/usr/bin/env python3
#
# get_ensemble_prediction.py
#
# MP Latham, 4/28/2026, script now saves std dev of ensemble predictions to npz file
#

import tensorflow as tf
import numpy as np
import sys, os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate ensemble predictions from trained models")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory containing the trained .keras models")
    parser.add_argument("-data_file", type=str, required=True, help="Path to the input data file (e.g., final_exam_dataset.npz)")
    parser.add_argument("-out_file", type=str, default="final_exam_results.npz", help="Path to save the output results npz file")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Load Data 
    print(f"Loading data from {args.data_file}...")
    data = np.load(args.data_file, allow_pickle=True) 
    features_list = data['features']
    true_labels_list = data['labels']

    # Load Models
    print(f"Discovering models in {args.model_dir}...")
    ensemble_models = []
    
    # Automatically find and load all .keras files in the directory
    for fname in sorted(os.listdir(args.model_dir)):
        if fname.endswith(".keras"):
            model_path = os.path.join(args.model_dir, fname)
            ensemble_models.append(tf.keras.models.load_model(model_path, compile=False))
            
    n_models = len(ensemble_models)
    if n_models == 0:
        print(f"Error: No '.keras' models were found in the directory '{args.model_dir}'.")
        return
        
    print(f"Successfully loaded {n_models} models.")

    # Get Predictions
    ensemble_predictions = []
    print("Generating predictions from the ensemble...")

    for features in tqdm(features_list):
        features_batch = np.expand_dims(features, axis=0).astype(np.float32)
        
        model_preds = []
        for model in ensemble_models:
            local_pred, global_pred = model.predict(features_batch, verbose=0)
            
            model_preds.append({
                "tauC_ns": global_pred.squeeze(),
                "local_params": local_pred.squeeze()
            })
        
        # Average the predictions across the ensemble
        avg_pred_tauC = np.mean([p['tauC_ns'] for p in model_preds])
        avg_pred_local = np.mean([p['local_params'] for p in model_preds], axis=0)

        std_pred_tauC = np.std([p['tauC_ns'] for p in model_preds])
        std_pred_local = np.std([p['local_params'] for p in model_preds], axis=0)
        
        ensemble_predictions.append({
            "tauC_ns": avg_pred_tauC,
            "tauC_err": std_pred_tauC,
            "S2": avg_pred_local[:, 1],
            "S2_err": std_pred_local[:, 1],
            "tauE_ns": avg_pred_local[:, 0],
            "tauE_err_ns": std_pred_local[:, 0],
            "Rex": avg_pred_local[:, 2],
            "Rex_err": std_pred_local[:, 2]
        })

    # Save Results
    # Make sure output directory exists if provided in out_file path
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or '.', exist_ok=True)
    
    np.savez_compressed(
        args.out_file,
        true_labels=true_labels_list,
        predicted_labels=np.array(ensemble_predictions, dtype=object)
    )
    print(f"\n'{args.out_file}' has been created.")

if __name__ == "__main__":
    main()

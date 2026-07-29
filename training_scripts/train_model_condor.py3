#!/usr/bin/env python3
"""
Modified from model_global_tauC_variable_length_v7.py3
    -For training single models using HTCondor

@author: MP Latham
@date: Feb. 22, 2026
"""

import sys, os
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalAveragePooling1D, LayerNormalization, Attention, Masking, MultiHeadAttention, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# ARGUMENT PARSER
def parse_args():
    parser = argparse.ArgumentParser(description="Train a single ModelFree AI Ensemble Member")
    parser.add_argument("-model_index", type=int, required=True, help="Index of the model (e.g., 0-9)")
    parser.add_argument("-train_dir", type=str, required=True, help="Directory containing TFRecord training data")
    parser.add_argument("-model_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("-batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-epochs", type=int, default=250, help="Maximum number of epochs")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return parser.parse_args()

# HARDWARE SETUP
mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# FUNCTIONS
def _parse_function(example_proto):
    """Decodes a single tf.train.Example message."""
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'local_labels': tf.io.FixedLenFeature([], tf.string),
        'global_label': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    features = tf.io.parse_tensor(parsed_example['features'], out_type=tf.float32)
    local_labels = tf.io.parse_tensor(parsed_example['local_labels'], out_type=tf.float32)
    global_label = parsed_example['global_label']

    # Reshape global label to be (1,) for consistency
    global_label = tf.reshape(global_label, [1])

    return features, {"local_output": local_labels, "global_output": global_label}

def count_records(file_list):
    count = 0
    for fn in file_list:
        # This iterates through the file and sums the records
        count += sum (1 for _ in tf.data.TFRecordDataset(fn))
    return count

def weighted_local_loss(y_true, y_pred):
    '''
    Custom loss to handle:
    1. Channel scaling (tauE is 1000x larger than S2)
    2. Rex sparsity (penalize missing an active Rex residue)
    '''
    # Calculate absolute error (MAE)
    error = tf.abs(y_true - y_pred)

    # Split into channels (tauE, S2, Rex)
    tauE_error = error[:, :, 0]
    S2_error = error[:, :, 1]
    Rex_error = error[:, :, 2]

    # 1. Scale Balancing
    # tauE is ~0-1 ns, S2 is 0-1, Rex is 1-25
    w_tauE = 10.0
    w_S2 = 10.0
    w_Rex = 0.4

    # 2. Rex sparsity
    # Identify where real exchange exists
    Rex_true = y_true[:, :, 2]
    # Apply 5x penalty if model misses a real exchange event
    sparsity_weights = tf.where(Rex_true > 0.1, 10.0, 1.0)

    # Combine terms
    total_loss = (w_tauE * tauE_error) + \
                 (w_S2 * S2_error) + \
                 (w_Rex * Rex_error * sparsity_weights)

    return total_loss

def build_lstm_multitask_model(n_features=3, return_attention=False):
    '''This is the ML model for variable-length sequences.'''
    # **VARIABLE LENGTH INPUT**
    inputs = Input(shape=(None, n_features), name='input_layer')

    # This creates a mask
    x = Masking(mask_value=-99.0)(inputs)

    # Shared layers work automatically with variable length
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(96, return_sequences=True))(x)
    x = Dense(96, activation='relu')(x)
    x = LayerNormalization()(x)

    # local output branch (tauE, S2, Rex)
    local_output = Dense(3, name='local_output', dtype='float32')(x)

    # Global pooling for tauC
    attention_layer = MultiHeadAttention(num_heads=2, key_dim=64, name='attention_layer')

    if return_attention:
        # Capture both the pooling sequence and the raw attention matrix
        attention_output, attention_scores = attention_layer(
                query=x, value=x, key=x, return_attention_scores=True
                )
    else:
        # Standard execution during training to save memory
        attention_output = attention_layer(query=x, value=x, key=x)

    pooled = GlobalAveragePooling1D()(attention_output)
    g = Dense(64, activation='relu')(pooled)
    g = Dropout(0.3)(g)
    global_output = Dense(1, name='global_output', dtype='float32')(g)

    if return_attention:
        # Cleanly name the 3rd output for SPINDLE
        attention_scores = Activation('linear', name='attention_scores')(attention_scores)
        model =  Model(inputs=inputs, outputs=[local_output, global_output, attention_scores])
    else:
        model = Model(inputs=inputs, outputs=[local_output, global_output])

    return model

def plot_and_save_learning_curves(history, model_index, output_dir):
    '''Plots ands aves the learning curves for a single model run.'''
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Loss
    ax1.plot(epochs, history_dict['loss'], 'bo-', label='TrainingLoss')
    ax1.plot(epochs, history_dict['val_loss'], 'ro-', label='ValidationLoss')
    ax1.set_title(f'Model {model_index+1} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MAE-based)')
    ax1.legend()
    ax1.grid(True)

    # Plot Global MAE (Mean Absolute Error)
    ax2.plot(epochs, history_dict['global_output_mae'], 'bo-', label='TrainingGlobalMAE')
    ax2.plot(epochs,history_dict['val_global_output_mae'], 'ro-', label='ValidationGlobalMAE')
    ax2.set_title(f'Model {model_index+1} - Global tauC MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE (ns)')
    ax2.legend()
    ax2.grid(True)

    # Save the figure
    plot_filename = os.path.join(output_dir, f"learning_curves_model_{model_index}.png")
    plt.savefig(plot_filename)
    plt.close() # Close the figure to free up memory


def main():
    args = parse_args()
    
    print(f"--- Starting Training for Model Index: {args.model_index} ---")
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load Data
    search_pattern = os.path.join(args.train_dir, "training_data-*.tfrecord")
    tfrecord_files = tf.io.gfile.glob(search_pattern)
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {args.train_dir}")

    # Each job will shuffle differently, creating a unique Train/Val split
    np.random.shuffle(tfrecord_files)
    val_split = max(1, round(0.1 * len(tfrecord_files)))
    val_files = tfrecord_files[:val_split]
    train_files = tfrecord_files[val_split:]

    print("Counting sample TFRecord files (this may take a moment)...")
    N_TRAIN_SAMPLES = count_records(train_files)
    N_VAL_SAMPLES = count_records(val_files)
    print(f"Found {N_TRAIN_SAMPLES} training samples and {N_VAL_SAMPLES} validation samples.")

    # Build Datasets
    padding_values = (
        tf.constant(-99.0, dtype=tf.float32),
        {
            "local_output": tf.constant(0.0, dtype=tf.float32),
            "global_output": tf.constant(0.0, dtype=tf.float32)
        }
    )

    train_dataset = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.padded_batch(args.batch_size,
                        padded_shapes=([None, 3], {"local_output": [None, 3], "global_output": [1]}),
                        padding_values=padding_values).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(val_files).map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.padded_batch(args.batch_size,
                        padded_shapes=([None, 3], {"local_output": [None, 3], "global_output": [1]}),
                        padding_values=padding_values).prefetch(tf.data.AUTOTUNE)

    # Build & Compile Model
    model = build_lstm_multitask_model(return_attention=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-4,
            clipnorm=0.5,
            gradient_accumulation_steps=4
            ),
        loss={'local_output': weighted_local_loss, 'global_output': 'mae'},
        metrics={'local_output':'mae', 'global_output':'mae'}
    )
    
    # Train
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)
    
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

    # Save Outputs
    model_path = os.path.join(args.model_dir, f"model_{args.model_index}.keras")
    model.save(model_path)
    plot_and_save_learning_curves(history, args.model_index, args.model_dir)
    print(f"Model {args.model_index} successfully saved to {model_path}")

if __name__ == "__main__":
    main()

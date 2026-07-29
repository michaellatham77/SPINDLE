#!/bin/bash

# Activate your specific Python environment
# If using a standard virtual environment (venv):
source ~/tf-218/bin/activate

# For TensorFlow to share the GPU politely before python boots
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Silence cuFFT factory warnings
export TF_CPP_MIN_LOG_LEVEL="2"

echo "--- GPU DIAGNOSTIC ---"
echo "Host Node: $(hostname)"
echo "Assigned GPU ID: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "----------------------"

# Run the python script
# The "$@" passes all the arguments from HTCondor directly to Python
python3 train_model_condor.py3 "$@"

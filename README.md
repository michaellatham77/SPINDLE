# SPINDLE: Deep Learning for NMR Relaxation Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.07.31.742133-blue)](https://www.biorxiv.org/content/10.64898/2026.07.31.742133v1)

**SPINDLE** (**SPIN** dynamics from **D**eep **L**earning **E**nsemble) is a multitask Bidirectional LSTM network with Multi-Head Self-Attention designed to rapidly predict Model-Free NMR relaxation parameters directly from backbone experimental data ($R_1$, $R_2$, and $\{^1\text{H}\}\text{--}^{15}\text{N}$ NOE) collected at a single static magnetic field.

---

## Key Features

* **Instant Model-Free Parameterization:** Avoids computationally expensive non-linear least-squares fitting or model-selection routines.
* **Multitask Prediction:** Simultaneously outputs per-residue local dynamics ($S^2$, $\tau_e$, and $R_{ex}$) alongside the global rotational correlation time ($\tau_c$).
* **Self-Attention Explainability:** Features a global branch equipped with Multi-Head Self-Attention (2 heads) to capture sequence-wide dynamic correlations and output per-residue importance weights.
* **Multi-Field Support:** Pre-trained models available across standard magnetic field strengths (500 MHz to 1100 MHz $^1\text{H}$ Larmor frequency).
* **Variable-Length Sequence Handling:** Built-in sequence masking allows seamless processing of proteins of varying lengths without artificial truncation.

---

## Architecture Overview

The network takes a variable-length sequence (length $L$) of relaxation parameters ($R_1, R_2, \text{NOE}$) and processes it through:
1. **Shared Trunk:** Two Bidirectional LSTM layers (128 and 96 units), a 96-unit Dense layer (ReLU), and a `LayerNormalization` step.
2. **Local Branch:** Dense layer mapping normalized sequence features directly to per-residue $S^2$, $\tau_e$ (ps), and $R_{ex}$ ($\text{s}^{-1}$).
3. **Global Branch:** Multi-Head Self-Attention (2 heads, key dimension 64) applied to sequence features, followed by 1D Global Average Pooling (GAP), a 64-unit Dense layer with Dropout (0.3), and a 1-unit Dense output for global $\tau_c$ (ns).

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/michaellatham77/SPINDLE.git
cd SPINDLE
```
### 2. Set up virtual environment
```
python3 -m venv spindle
source spindle/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
## Quickstart & Usage

Input Data Format
Input files should be formatted as TSV files containing per-residue relaxation data. Unmeasured or missing values should be indicated with -99.0 (the default masking value).

|Residue|R1 (s^-1)|R2 (s^-1)|NOE|
|-------|---------|---------|---|
|1|1.25|8.40|0.78|
|2|1.30|8.92|0.81|
|...|...|...|...|

## Running SPINDLE via Command Line

Run SPINDLE on a sample dataset using the pre-trained weights for your specific spectrometer field strength:
```bash
./spindle.py3 -i example_data/example_600MHz_data.txt \
    -f 600 \
    -o example_600MHz \
    -m models \
    -p pdf
```
For explainability (i.e., attention scores) mapped on to predicted $S^2$ and $R_{ex}$ add ```-e``` argument to the above command.

## Running SPINDLE via Juypter notebook
Coming soon!


## Repository Structure
```Plaintext
SPINDLE/
├── spindle.py3                   # Command-line interface
├── spindle_calibrations.py       # Field-dependent linear calibration for \tau_c and error multipliers
├── relaxation_rates.py           # Spectral density function and relaxation rates definition
├── models/                       # Field-dependent models (ensemble Keras files per field strength)
├── training_scripts/
│   ├── calibration/              # Field-dependent validation and calibration scripts & ground truth
│   ├── train_ensemble.sub        # Submission file for HT-Condor
│   ├── test_model.py3            # Script to test DNN
│   ├── make_training_data.py3    # Script to generate training data
│   ├── train_model_condor.py3    # Script to train DNN
│   ├── run_condor_training.sh    # Shell script used by train_ensemble.sub to execute training
│   └── relaxation_rates.py       # Spectral density function and relaxation rates definitions
├── example_data/                 # Scripts and sample outputs (600 & 800 MHz data used for Fig S8)
├── for_manuscript/               # Synthetic & experimental evaluation datasets
│   ├── model_free/               # Synthetic data used for ModelFree comparison (Figs 3–5, S1–S3)
│   └── BMRB_data/                # Experimental BMRB data (Figs 6–7, S5–S7)
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use SPINDLE in your research, please cite our manuscript:
```bibtex
@article{krise2026,
    title={SPINDLE: Unlocking protein dynamics from single-field NMR relaxation at a using a deep learning ensemble},
    author{Olivia E Krise and Michael P Latham},
    journal={bioRxiv},
    year={2026},
    doi={https://doi.org/10.64898/2026.07.31.742133}
}
```
## Contact and Support

For questions, feature requests, or reporting issues, please open a GitHub issue or contact  
[Michael Latham](latha070@umn.edu)  
[Latham Lab](https://www.lathamlaboratory.org)

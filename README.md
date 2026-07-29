# SPINDLE: Deep Learning for NMR Relaxation Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41592--000--00000--0-blue)](https://doi.org)

**SPINDLE** (**S**equence-based **P**rediction of **I**nternal and **N**uclear **D**ynamics via **L**earning **E**mbeddings) is a multitask Bidirectional LSTM network with Multi-Head Self-Attention designed to rapidly predict Model-Free NMR relaxation parameters directly from backbone experimental data ($R_1$, $R_2$, and $\{^1\text{H}\}\text{--}^{15}\text{N}$ NOE).

---

## Key Features

* **Instant Model-Free Parameterization:** Avoids computationally expensive non-linear least-squares fitting or model-selection routines (e.g., *FAST-ModelFree*).
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
git clone [https://github.com/](https://github.com/)[your-username]/spindle-nmr.git
cd spindle-nmr

### 2. Clone the Repository
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

## Quickstart & Usage

Input Data Format
Input files should be formatted as CSV or TSV files containing per-residue relaxation data. Unmeasured or missing values should be indicated with -99.0 (the default masking value).
Residue	R1 (s^-1)	R2 (s^-1)	NOE
1	1.25	8.40	0.78
2	1.30	8.92	0.81
...	...	...	...

## Running SPINDLE via Command Line

Run SPINDLE on a sample dataset using the pre-trained weights for your specific spectrometer field strength:
```bash
./spindle.py3 -i example_data/example_600MHz_data.txt \
    -f 600 \
    -o example_600MHz \
    -m models \
    -p pdf

## Running SPINDLE via Juypiter notebook



## Repository Structure

SPINDLE/
|-spindle.py3                   # Command-line interface
|-spindle_calibrations.py       # Field dependent linear calibration for $\tau_c$ and error multipliers
|-relaxation_rates.py           # Spectral density function and relaxation rates definition
|-models/                       # Directories of field dependent models. Each field dependent sub-directory contains ten keras files for the ten models that make up the DNN ensemble
|-training_data/                # Directories of field dependent training data. Each field dependent sub-directory contains 40 tfrecord files of 10k synthetic 'proteins' each
|-training_scripts/
|   |-calibration/              # Directories of field dependent validation and calibration. Subdirectories contain field dependent validation data and ground truth. Main directory contains scripts
|                                 for generating validation data (generate_final_exam.py3), running SPINDLE on this (get_ensemble_predictions.py3), and analyzing the results (analyze_and_correct.py3).
                                  500 MHz data was used to generate Fig 2 of manuscript
|   |-train_ensemble.sub        # Submission file for HT-Condor
|   |-test_model.py3            # Script to test DNN
|   |-make_training_data.py3    # Script to make training data
|   |-train_model_condor.py3    # Script to train DNN
|   |-run_condor_training.sh    # Shell script used by train_ensemble.sub to execute training
|   |-relaxation_rates.py       # Spectral density function and relaxation rates definitions
|-example_data/                 # Scripts to generate and analyze example data. Two example outputs (600 and 800 MHz data) are included, and these data were used to generate Fig S8.
|-for_manuscript/               # Directory of synthetic data used for modelfree/fast-modelfree and BMRB analysis
|   |-model_free/               # Directory of synthetic data used for modelfree. These data were used to generate Figs 3-5 and Figs S1-S3 in manuscript.
|   |-BMRB_data/                # Directory of experimental BMRB data. These data were used to generate Figs 6 & 7 and Figs S5-57.
|-requirement.txt
|-LICENSE
|-README.md

## Citation

If you use SPINDLE in your research, please cite our manuscript:
@article{[],
    title={},
    author{[Olivia Krise and Michael P Latham]},
    journal={[]},
    year={[2026]},
    doi={}
}

## Contact and Support

For questions, feature requests, or reporting issues, please open a GitHub issue or contact
[Michael Latham] - latha070@umn.edu
[Latham Lab] - https://www.lathamlaboratory.org

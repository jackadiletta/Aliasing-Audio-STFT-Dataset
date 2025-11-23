🔊 Aliasing End-to-End
Dataset Construction • Aliasing Classifiers • LSTM Grid Search • Downstream Task Difficulty

This repository contains a complete pipeline for constructing aliasing-controlled audio datasets, training binary aliasing detectors, and evaluating how aliasing affects downstream sound-classification tasks.

It provides:

Deterministic STFT and MFCC feature datasets

Paired filtered (anti-aliased) and unfiltered (aliased) examples

ML classifiers (XGBoost, AdaBoost) for aliasing detection

LSTM-based grid search for aliasing presence prediction

CNN/LSTM downstream models for class label prediction on filtered vs. unfiltered sets

Difficulty analysis across six sample rates (1 kHz → 22 kHz)

📁 Dataset Construction
Source Corpora

UrbanSound8K (U8K)

ESC-50

MAVD (Zenodo city-sound dataset)

Each audio file is processed into 1-second chunks and transformed into aliased / anti-aliased pairs across:

[ 1000, 2000, 4000, 8000, 16000, 22000 ] Hz

🔧 Processing Overview (from process.ipynb)
1. Aliasing Manipulation

Unfiltered (aliased)

raw = resample_poly(chunk, sr, fs_original)


Filtered (anti-aliased)

sos = butter(6, cutoff=sr/2, fs=fs_original, btype='low', output='sos')
filtered  = sosfilt(sos, chunk)
filt_rs   = resample_poly(filtered, sr, fs_original)

2. Feature Extraction
STFT features

STFT computed after resampling using:

n_fft = 512

hop = 256

Magnitudes are globally normalized using two-pass min-max across all bins.

MFCC features

n_mfcc = 20

n_fft = sr // 2, hop = sr // 4

All MFCC vectors are z-scored using train-split statistics only.

Saved stats:

mfcc_norm_stats.npz  → { mean, std }

📂 Dataset Directory Structure
Processed_Files/
    DS_U8K/               # UrbanSound8K
    DS_ESC/               # ESC-50
    DS_ZEN/               # MAVD (no class labels → class 0)
        22000/
        16000/
        8000/
        4000/
        2000/
        1000/
            train/
                filtered/      # .npy feature files
                unfiltered/
            validation/
                filtered/
                unfiltered/
            test/
                filtered/
                unfiltered/


File naming convention:

<index>-<class>.npy


Where <class> comes from the original dataset.
MAVD uses "0" because there are no class labels.

🧪 Aliasing Presence Classifiers

(Included in process.ipynb)

The classical aliasing classifiers operate on flattened STFT/MFCC vectors.

Models include:

XGBoost

AdaBoost(RandomForest)

AdaBoost(DecisionTree)

Each model is trained per sample rate, per feature type, per dataset.

Outputs:

Pickled models (models/*.pkl)

CSV performance report:

model_performance_<timestamp>.csv

🧠 LSTM Aliasing Classifier (Grid Search)

(From LSTM_GS.ipynb)

The LSTM classifier performs binary detection:

0 → unfiltered (aliased)
1 → filtered   (anti-aliased)

Pipeline:

Loads .npy sequences (STFT or MFCC)

Zero-pads variable-length sequences

Runs grid search over:

LSTM hidden dimension

Fully-connected dimension

Key command example:

grid_search(root_dir, sr=8000)


Runs grid search and prints F1 scores for each configuration.

🎯 Downstream Task Difficulty

(From Difficulty_Separate.ipynb)

This experiment trains class label predictors (ESC-50 or U8K classes), but using only filtered or only unfiltered data.

This answers:
“How much does aliasing harm (or help) downstream environmental sound classification?”

Models:

LSTMClassifier

CNNClassifier

Each trained per:

Feature type (STFT or MFCC)

Sample rate (1 kHz → 22 kHz)

Subset (filtered only OR unfiltered only)

Outputs:

difficulty_separate_by_subset_esc.csv

📊 Reproducibility & Normalization

1-second chunking

STFT normalization: global min-max

MFCC normalization: per-dimension z-score (train-split stats)

GPU-safe checkpointing for downstream tasks

Deterministic random seeds in LSTM grid search

🚀 Installation
pip install numpy scipy librosa soundfile tqdm pandas scikit-learn xgboost torch matplotlib


For GPU acceleration:

https://pytorch.org/get-started/locally/

🧭 Usage Summary
Build datasets:

Run process.ipynb (STFT & MFCC).

Classical aliasing classifiers:

Also inside process.ipynb.

LSTM aliasing grid search:

Run cells inside LSTM_GS.ipynb (per sample rate).

Downstream task difficulty:

Run Difficulty_Separate.ipynb.

📌 Citation

If you use this repository, please cite:

UrbanSound8K

ESC-50

MAVD / Zenodo city-sound dataset

This repository (link to your GitHub page)

📝 License

Add your license here (e.g., MIT License).

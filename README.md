Aliasing End-to-End — Dataset, Classifiers, and Downstream Tasks

This repository accompanies a study of aliasing in environmental audio. It provides:

A deterministic dataset builder that produces paired unfiltered (aliased) and filtered (anti-aliased) versions of 1-second chunks at multiple target sample rates.

Aliasing presence classifiers (traditional ML + LSTM grid search) that predict whether a chunk contains aliasing.

Downstream task analysis that measures how aliasing impacts standard audio classification (class label prediction) when models train/test on only filtered vs only unfiltered data.

Contents

Data processing & classical aliasing ML: /mnt/data/process.ipynb

Builds STFT/MFCC datasets and runs classical ML aliasing classifiers (XGBoost / AdaBoost ensembles).

LSTM aliasing grid search: /mnt/data/LSTM_GS.ipynb

End-to-end LSTM grid search for binary aliasing detection across sample rates.

Downstream task analysis: /mnt/data/Difficulty_Separate.ipynb

Trains LSTM/CNN on class labels using only filtered or only unfiltered data to quantify difficulty.

(Optional, combined views): /mnt/data/Difficulty_Analysis_Joined.ipynb

The notebooks above mirror the Python snippets shown in this README.

1) Dataset Construction
Source corpora

UrbanSound8K, ESC-50, MAVD (Zenodo city-sound)
We read each file, take exact 1-second chunks, and generate paired training examples across target sampling rates:

target_rates = [1000, 2000, 4000, 8000, 16000, 22000]

Aliasing manipulation (core function)

Unfiltered (aliased): resample_poly(chunk, sr, fs)

Filtered (anti-aliased): low-pass at Nyquist(sr) in the original sampling domain, then resample

sos = butter(6, cutoff=sr/2, fs=fs, btype='low', output='sos')
filtered = sosfilt(sos, chunk)
filt_rs  = resample_poly(filtered, sr, fs)


Chunk selection: Optional significance gating to keep chunks with sufficient broadband content (mean-square power and high-frequency energy above thresholds).

Features

STFT (image-like)

n_fft=512, hop=n_fft/2, computed after resampling with fs=sr

Two-pass global min–max across the entire corpus; saved spectrograms are in [0,1].

MFCC (feature-vector–like)

n_mfcc=20, analysis tied to the target sr (n_fft=sr//2, hop=sr//4)

Z-score per-dimension using train-split statistics only; saved features are already standardized.

Normalization stats saved as mfcc_norm_stats.npz (keys: mean, std).

Directory layout & naming
Processed_Files/
    DS_U8K/                # (also DS_ESC, DS_ZEN)
        22000/
        16000/
        8000/
        4000/
        2000/
        1000/
            train/
                filtered/      # .npy files
                unfiltered/
            validation/
                filtered/
                unfiltered/
            test/
                filtered/
                unfiltered/


STFT files: each .npy is a min–max normalized magnitude spectrogram (no dB in saved file).

MFCC files: each .npy is a flattened & z-scored feature vector.

Filenames: <index>-<class>.npy where <class> comes from the source corpus filename; MAVD uses 0 (no class).

2) Aliasing Presence Classifiers (Binary: filtered vs unfiltered)

You’ll find two implementations:

A) Classical ML (in process.ipynb)

Loads flattened features per rate/split from the dataset folders above.

Balances the classes (filtered vs unfiltered) by downsampling.

Trains & evaluates:

XGBoost (with a lightweight grid on max_depth, learning_rate)

AdaBoost(RandomForest), AdaBoost(DecisionTree)

Writes per-rate validation metrics to a timestamped CSV.

Entry points (illustrative):

SAMPLE_RATES = [1000, 2000, 4000, 8000, 16000, 22000]
DATASETS = {
    'STFT': [
        r"D:\Aliasing3\Processed_Files\DS_U8K",
        r"D:\Aliasing3\Processed_Files\DS_ZEN"
    ],
    # 'MFCC': [...]
}


If mfcc_norm_stats.npz exists in a dataset root, the loader applies the saved z-score at load time (a safety check; MFCC files are already normalized).

Outputs

Trained model pickles per rate: models/{name}_{sr}Hz.pkl

CSV summary: model_performance_<timestamp>.csv

B) LSTM Grid Search (in LSTM_GS.ipynb)

Builds a binary classifier: 0=unfiltered vs 1=filtered

Sequence input: (time_steps, feat_dim) from either STFT (2D) or MFCC (reshaped from flat).

Collates variable-length sequences with zero-padding.

Grid search over LSTM hidden_dim and fc_dim per rate.

Key bits

# Labeling by folder name
for label, cls in [('unfiltered', 0), ('filtered', 1)]:
    folder = os.path.join(root_dir, str(sr), split, label)
    # collect files and labels...


Outputs

Best F1 per configuration, printed to console (CSV save is scaffolded in the code).

Use this to compare how strongly aliasing signatures are learnable at each sample rate.

3) Downstream Task Analysis (Difficulty of classifying sound events)

Goal: quantify how aliasing affects standard classification (e.g., ESC-50 class IDs).
We train on only filtered or only unfiltered subsets and predict the class label, not aliasing.

Notebook: /mnt/data/Difficulty_Separate.ipynb

What it does

Loops over:

Feature type: STFT or MFCC

Dataset root (e.g., DS_ESC, DS_U8K, …)

Sample rates: 1k → 22k

Model type: LSTM / CNN

Subset: filtered or unfiltered (one at a time)

Labels are extracted from the filenames: <index>-<class>.npy.

For each config:

Train for epochs with best-val checkpointing (saves each epoch, reloads best).

Evaluate on the matching test split (filtered only or unfiltered only).

Log best_val_acc and test_acc.

Models

LSTMClassifier: 3-layer LSTM → last hidden → FC → num_classes

CNNClassifier: 3 conv layers + adaptive global pooling → MLP head
(expects input as (B, T, D); internally permutes to (B, 1, D, T))

Output

CSV: difficulty_separate_by_subset_esc.csv (per-config test accuracy).

Interpretation: If accuracy drops on a given subset (e.g., unfiltered), aliasing harms the downstream task at that rate/feature.

4) Reproducibility & Normalization

Chunking: exact 1-second windows (first pass may discard windows that fail significance gates).

STFT: n_fft=512, hop=256, computed after resampling with fs=sr.

Dataset saves magnitudes with global min–max scaling.

Visualizations may display dB for readability (this is only for plotting).

MFCC: n_mfcc=20, n_fft=sr//2, hop=sr//4; Mel filterbank implicitly capped by sr/2.

Saved features are z-scored using train-split statistics (no leakage).

Stats stored at the dataset root: mfcc_norm_stats.npz.

5) Installation
# (Recommended) Create a virtual environment first
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

pip install -U pip
pip install numpy scipy librosa soundfile tqdm matplotlib pandas scikit-learn xgboost torch torchvision torchaudio


GPU (optional): install CUDA-enabled PyTorch per the official selector
.

6) Quickstart
A) Build datasets (STFT & MFCC)

Open /mnt/data/process.ipynb
 and run the cells that:

Collect file lists (UrbanSound8K, ESC-50, MAVD).

(STFT) Two-pass global min–max with find_global_bounds.

Save: build_and_save_dataset(...) → Processed_Files/DS_*/*/split/(filtered|unfiltered)/...

(MFCC) Save via create_mfcc_dataset(...) (z-score saved).

B) Aliasing presence (classical ML)

In the same notebook, run the modeling cells:

Select dataset roots in DATASETS

Run the training loop to produce validation metrics and model pickles.

C) Aliasing presence (LSTM grid search)

Open /mnt/data/LSTM_GS.ipynb
:

Set root to one dataset root (e.g., DS_ESC_MFCC or DS_ZEN).

Run grid_search(root, sr=...) for each sample rate.

D) Downstream task difficulty

Open /mnt/data/Difficulty_Separate.ipynb
:

Configure DATASETS, frequencies, subsets, epochs.

Run the experiment loop; results go to difficulty_separate_by_subset_esc.csv.

7) Tips & Conventions

Color scale consistency (plots)

STFT (dB): fix color limits, e.g., vmin=-80, vmax=0 for truthful comparisons across panels.

MFCC: choose a fixed range or compute global min/max across panels before plotting.

Exact method match

STFT: leave boundary at SciPy’s default (zero-pad) to match the dataset builder.

MFCC: use exactly n_fft=sr//2, hop=sr//4 to mirror the saved features.

MAVD classes

MAVD is not class-labeled; files are saved with class 0. Use MAVD primarily for aliasing presence, not downstream class accuracy.

8) Results & Reporting

Aliasing presence: report metrics per rate (1k→22k) to show where aliasing is most detectable.

Downstream difficulty: compare test accuracy for filtered-only vs unfiltered-only at each rate/feature to quantify aliasing impact on real tasks.

9) Citation

If you use this pipeline or datasets in academic work, please cite the repository and the original source datasets (UrbanSound8K, ESC-50, MAVD).


Questions / Issues

Open an issue with:

dataset root path(s) used,

feature type (STFT vs MFCC),

sample rate(s),

exact error trace (if any).

Happy aliasing hunting 🔍🎧

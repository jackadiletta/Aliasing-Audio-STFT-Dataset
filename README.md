# Aliasing End-to-End

A streamlined toolkit for studying aliasing in environmental audio. The repository provides:

## Key Components
- **Deterministic dataset builder** generating paired *unfiltered (aliased)* and *filtered (anti-aliased)* 1-second audio chunks across multiple sample rates (1k–22k Hz).
- **Aliasing presence classifiers** using classical ML (XGBoost, AdaBoost) and LSTM models to distinguish filtered vs unfiltered audio.
- **Downstream task evaluation** measuring how aliasing affects standard audio classification accuracy when models train/test on only filtered or only unfiltered data.

## Dataset Structure
- Supports major audio corpora: UrbanSound8K, ESC-50, MAVD.
- Produces STFT or MFCC feature datasets with consistent normalization:
  - STFT: global min–max-scaled magnitude spectrograms  
  - MFCC: z-scored features using train-set statistics
- Organized as:
  - Processed_Files/DS_* / <sr> / (train|validation|test) / (filtered|unfiltered) / file.npy


## Models
- **Aliasing detection:** binary classifiers determining if a chunk contains aliasing.
- **Downstream classifiers:** LSTM and CNN architectures predicting sound class labels, evaluated separately on filtered and unfiltered subsets.

## Notebooks
- **process.ipynb:** dataset construction + classical ML aliasing models  
- **LSTM_GS.ipynb:** LSTM grid search for aliasing detection  
- **Difficulty_Separate.ipynb:** downstream class-label difficulty experiments  
- **Difficulty_Analysis_Joined.ipynb:** optional combined views

## Outputs
- Trained model files  
- CSV logs of detection accuracy and downstream task performance  
- Fully reproducible datasets and normalization statistics

## Purpose
To quantify:
1. How detectable aliasing is at different sample rates.  
2. How aliasing affects real-world audio classification tasks.

## Citation
Please cite this repository and the source datasets (UrbanSound8K, ESC-50, MAVD) in academic work.

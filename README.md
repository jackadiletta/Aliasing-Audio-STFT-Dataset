# Publications and Impact:

This repository accompanies two complementary research efforts that together establish a foundation for aliasing-aware audio processing in low-power, always-on IoT sensing.

**Dataset Paper** – “A Novel Aliasing vs Non-Aliasing Audio Dataset for Always-On IoT Microphone Experimentation”
https://dl.acm.org/doi/abs/10.1145/3736425.3771962

This work introduces the first curated dataset of deliberately aliased audio, constructed from three standard environmental sound datasets and resampled across multiple sub-Nyquist rates. Because IoT microphones increasingly rely on low sampling rates to reduce power and bandwidth, aliasing becomes unavoidable and can significantly impact downstream tasks such as audio classification. The dataset enables researchers to (1) systematically study aliasing effects, (2) benchmark robustness of audio models under degraded sampling, and (3) develop classifiers that detect whether audio exhibits aliasing.

**System Paper** – “EfficientMic: Adaptive Acoustic Sensing with a Single Microphone for Smart Infrastructure”
https://dl.acm.org/doi/epdf/10.1145/3736425.3770108

This work presents EfficientMic, a fully digital, aliasing-aware adaptive sampling framework that uses only a single microphone—no analog filters or multi-mic arrays. EfficientMic detects aliasing in real time using lightweight STFT-based features and an XGBoost classifier, dynamically tuning the sampling rate to minimize energy and storage while preserving task performance. Deployments on a Raspberry Pi demonstrate 70%+ storage savings and 50%+ energy reduction with near-baseline accuracy, highlighting a practical, hardware-independent path toward scalable low-power acoustic sensing.

Together, these deliverables provide both the data resources and the algorithmic framework needed to build next-generation, aliasing-aware IoT audio systems.

# Aliasing End-to-End

A streamlined toolkit for studying aliasing in environmental audio. The repository provides:

## Key Components
- **Deterministic dataset builder** generating paired *unfiltered (aliased)* and *filtered (anti-aliased)* 1-second audio chunks across multiple sample rates (1k–22k Hz).
- **Aliasing presence classifiers** using classical ML (XGBoost, AdaBoost) and LSTM models to distinguish filtered vs unfiltered audio.
- **Downstream task evaluation** measuring how aliasing affects standard audio classification accuracy when models train/test on only filtered or only unfiltered data.

## Dataset Structure
- Supports major audio corpora: UrbanSound8K, ESC-50, MAVD.
- Produces STFT or MFCC feature datasets with consistent normalization:
  - STFT (short time fourier transform): global min–max-scaled magnitude spectrograms  
  - MFCC (Mel-frequency Cepstral Coefficients): z-scored features using train-set statistics
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

# Sound Data Collection and Processing

This project implements Part 3 of the User Identity and Product Recommendation System assignment: **Sound Data Collection and Processing**.

## Overview

This module processes audio samples, applies augmentations, and extracts features for voice validation in the multimodal authentication system.

## Features

- ✅ Loads audio samples (supports .m4a, .wav, .mp3 formats)
- ✅ Displays waveforms and spectrograms for visualization
- ✅ Applies multiple augmentations per sample:
  - Pitch shift
  - Time stretch
  - Background noise addition
- ✅ Extracts audio features:
  - MFCCs (13 coefficients)
  - Spectral roll-off
  - Energy (RMS)
- ✅ Saves features to `audio_features.csv`

## Files

- `audio_processing.py` - Standalone Python script for processing
- `audio_processing.ipynb` - Jupyter notebook with detailed visualizations
- `requirements.txt` - Required Python packages
- `Audio samples/` - Directory containing audio files

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Python Script

```bash
python audio_processing.py
```

This will:
- Process all audio files in the `Audio samples/` directory
- Display waveforms and spectrograms
- Apply augmentations
- Extract features
- Save results to `audio_features.csv`

### Option 2: Use Jupyter Notebook

1. Open `audio_processing.ipynb` in Jupyter
2. Run all cells sequentially
3. View visualizations and analysis inline

## Audio Files

Place your audio samples in the `Audio samples/` directory. The script supports:
- `.m4a` files
- `.wav` files
- `.mp3` files

Expected format: At least 2 audio samples with phrases like:
- "Yes, approve"
- "Confirm transaction"

## Output

The script generates `audio_features.csv` with the following columns:
- `label`: Audio sample label (filename)
- `augmentation`: Type of augmentation applied (original, pitch_shift, time_stretch, add_noise)
- `sample_rate`: Audio sample rate
- `duration`: Audio duration in seconds
- `energy`: RMS energy
- `spectral_rolloff`: Spectral roll-off frequency
- `mfcc_1` through `mfcc_13`: MFCC coefficients

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- librosa >= 0.9.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- soundfile >= 0.10.0

## Notes

- Audio files are automatically resampled to 22050 Hz for consistency
- All audio is converted to mono channel
- Each original sample generates 4 feature vectors (original + 3 augmentations)







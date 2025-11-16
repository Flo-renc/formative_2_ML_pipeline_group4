"""
Sound Data Collection and Processing
This script processes audio samples, applies augmentations, and extracts features.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path


class AudioProcessor:
    """Class to handle audio processing, augmentation, and feature extraction."""
    
    def __init__(self, audio_dir="Audio samples", sample_rate=22050):
        """
        Initialize the AudioProcessor.
        
        Parameters:
        -----------
        audio_dir : str
            Directory containing audio files
        sample_rate : int
            Target sample rate for audio processing (default: 22050)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.features_list = []
        
    def load_audio(self, filepath):
        """
        Load audio file and convert to mono if needed.
        
        Parameters:
        -----------
        filepath : str
            Path to audio file
            
        Returns:
        --------
        y : np.ndarray
            Audio time series
        sr : int
            Sample rate
        """
        try:
            # Load audio file (handles .wav, .mp3, etc.)
            y, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            return y, sr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def plot_waveform(self, audio, sr, title="Waveform", save_path=None):
        """
        Plot waveform of audio signal.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        title : str
            Plot title
        save_path : str
            Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sr, alpha=0.8)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_spectrogram(self, audio, sr, title="Spectrogram", save_path=None):
        """
        Plot spectrogram of audio signal.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        title : str
            Plot title
        save_path : str
            Path to save the plot (optional)
        """
        # Compute short-time Fourier transform (STFT)
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', 
                                 hop_length=512, cmap='viridis')
        plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def augment_pitch_shift(self, audio, sr, n_steps=2):
        """
        Apply pitch shift augmentation.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        n_steps : int
            Number of semitones to shift (positive = higher, negative = lower)
            
        Returns:
        --------
        augmented : np.ndarray
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def augment_time_stretch(self, audio, rate=1.2):
        """
        Apply time stretch augmentation.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        rate : float
            Stretch factor (>1.0 = slower, <1.0 = faster)
            
        Returns:
        --------
        augmented : np.ndarray
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def augment_add_noise(self, audio, noise_factor=0.005):
        """
        Add background noise to audio.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        noise_factor : float
            Amplitude of noise relative to signal
            
        Returns:
        --------
        augmented : np.ndarray
            Audio with added noise
        """
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented
    
    def extract_mfcc(self, audio, sr, n_mfcc=13):
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        n_mfcc : int
            Number of MFCC coefficients to extract
            
        Returns:
        --------
        mfcc : np.ndarray
            MFCC features (mean across time)
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Return mean across time
    
    def extract_spectral_rolloff(self, audio, sr, roll_percent=0.85):
        """
        Extract spectral roll-off feature.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        roll_percent : float
            Roll-off percentage (default: 0.85)
            
        Returns:
        --------
        rolloff : float
            Spectral roll-off frequency
        """
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent)
        return np.mean(rolloff)
    
    def extract_energy(self, audio):
        """
        Extract energy feature (RMS energy).
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
            
        Returns:
        --------
        energy : float
            RMS energy
        """
        rms = librosa.feature.rms(y=audio)
        return np.mean(rms)
    
    def extract_all_features(self, audio, sr, label="", augmentation_type=""):
        """
        Extract all features from audio signal.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio time series
        sr : int
            Sample rate
        label : str
            Label for the audio sample
        augmentation_type : str
            Type of augmentation applied (if any)
            
        Returns:
        --------
        features : dict
            Dictionary containing all extracted features
        """
        # Extract MFCCs (13 coefficients)
        mfcc = self.extract_mfcc(audio, sr, n_mfcc=13)
        
        # Extract spectral roll-off
        rolloff = self.extract_spectral_rolloff(audio, sr)
        
        # Extract energy
        energy = self.extract_energy(audio)
        
        # Create feature dictionary
        features = {
            'label': label,
            'augmentation': augmentation_type,
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'energy': energy,
            'spectral_rolloff': rolloff
        }
        
        # Add MFCC coefficients
        for i in range(len(mfcc)):
            features[f'mfcc_{i+1}'] = mfcc[i]
        
        return features
    
    def process_all_samples(self):
        """
        Process all audio samples: load, visualize, augment, and extract features.
        """
        audio_files = list(Path(self.audio_dir).glob("*.m4a"))
        audio_files.extend(list(Path(self.audio_dir).glob("*.wav")))
        audio_files.extend(list(Path(self.audio_dir).glob("*.mp3")))
        
        if not audio_files:
            print(f"No audio files found in {self.audio_dir}")
            return
        
        print(f"Found {len(audio_files)} audio file(s)")
        print("=" * 60)
        
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            print("-" * 60)
            
            # Load original audio
            audio, sr = self.load_audio(str(audio_file))
            if audio is None:
                continue
            
            label = audio_file.stem  # Get filename without extension
            
            # Display original waveform
            print(f"Displaying waveform for: {label}")
            self.plot_waveform(audio, sr, title=f"Waveform - {label}")
            
            # Display original spectrogram
            print(f"Displaying spectrogram for: {label}")
            self.plot_spectrogram(audio, sr, title=f"Spectrogram - {label}")
            
            # Extract features from original
            print(f"Extracting features from original: {label}")
            features_original = self.extract_all_features(audio, sr, label=label, 
                                                         augmentation_type="original")
            self.features_list.append(features_original)
            
            # Apply augmentations
            print(f"\nApplying augmentations to: {label}")
            
            # Augmentation 1: Pitch shift
            print("  - Applying pitch shift (n_steps=2)...")
            audio_pitch = self.augment_pitch_shift(audio, sr, n_steps=2)
            features_pitch = self.extract_all_features(audio_pitch, sr, label=label,
                                                      augmentation_type="pitch_shift")
            self.features_list.append(features_pitch)
            
            # Augmentation 2: Time stretch
            print("  - Applying time stretch (rate=1.2)...")
            audio_time = self.augment_time_stretch(audio, rate=1.2)
            features_time = self.extract_all_features(audio_time, sr, label=label,
                                                     augmentation_type="time_stretch")
            self.features_list.append(features_time)
            
            # Augmentation 3: Add noise (bonus - more than required)
            print("  - Adding background noise...")
            audio_noise = self.augment_add_noise(audio, noise_factor=0.005)
            features_noise = self.extract_all_features(audio_noise, sr, label=label,
                                                      augmentation_type="add_noise")
            self.features_list.append(features_noise)
            
            print(f"✓ Completed processing: {label}\n")
        
        print("=" * 60)
        print("All samples processed successfully!")
    
    def save_features_to_csv(self, output_file="audio_features.csv"):
        """
        Save extracted features to CSV file.
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        if not self.features_list:
            print("No features to save. Please process audio samples first.")
            return
        
        df = pd.DataFrame(self.features_list)
        df.to_csv(output_file, index=False)
        print(f"\nFeatures saved to {output_file}")
        print(f"Total samples: {len(df)}")
        print(f"Features shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        return df


def main():
    """Main function to run the audio processing pipeline."""
    # Initialize processor
    processor = AudioProcessor(audio_dir="Audio samples", sample_rate=22050)
    
    # Process all audio samples
    processor.process_all_samples()
    
    # Save features to CSV
    processor.save_features_to_csv("audio_features.csv")
    
    print("\n" + "=" * 60)
    print("Audio processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


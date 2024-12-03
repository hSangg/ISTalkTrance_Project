import numpy as np
import librosa
import scipy.signal as signal
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import soundfile as sf
import os
from scipy.ndimage import gaussian_filter1d
from flask import Flask, request, jsonify

app = Flask(__name__)

class AutoSpectralDiarizer:
    def __init__(self, min_speakers=2, max_speakers=8):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
    
    def load_audio(self, audio_path, target_sr=16000):
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio, sr
    
    def extract_features(self, audio, sr):
        hop_length = 512
        n_fft = 2048
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128
        )
        
        mfccs = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec),
            n_mfcc=20
        )
        
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=hop_length
        )
        
        rms = librosa.feature.rms(
            y=audio,
            hop_length=hop_length
        )
        
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        features = np.vstack([
            mfccs,
            spectral_centroids,
            spectral_bandwidth,
            spectral_rolloff,
            zcr,
            rms
        ])
        
        return features, hop_length
    
    def estimate_number_of_speakers(self, features):
        X = features.T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        energy = np.mean(X_scaled, axis=1)
        voice_mask = energy > np.percentile(energy, 10)
        X_voiced = X_scaled[voice_mask]
        
        best_score = -1
        best_n_speakers = self.min_speakers
        
        print("Estimating number of speakers...")
        
        for n_speakers in range(self.min_speakers, self.max_speakers + 1):
            clustering = SpectralClustering(
                n_clusters=n_speakers,
                random_state=42,
                n_init=5,
                affinity='nearest_neighbors'
            )
            
            try:
                labels = clustering.fit_predict(X_voiced)
                score = silhouette_score(X_voiced, labels)
                print(f"Testing {n_speakers} speakers - Silhouette score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_n_speakers = n_speakers
            except Exception as e:
                print(f"Failed to cluster for {n_speakers} speakers: {e}")
                continue
        
        print(f"\nOptimal number of speakers detected: {best_n_speakers}")
        return best_n_speakers
    
    def segment_speakers(self, features, n_speakers):
        X = features.T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clustering = SpectralClustering(
            n_clusters=n_speakers,
            random_state=42,
            n_init=5,
            affinity='nearest_neighbors'
        )
        
        labels = clustering.fit_predict(X_scaled)
        smoothed_labels = gaussian_filter1d(labels, sigma=3)
        smoothed_labels = np.round(smoothed_labels).astype(int)
        
        return smoothed_labels
    
    def separate_speakers(self, audio, sr, labels, hop_length):
        speaker_audio = {}
        frame_length = hop_length
        
        window = signal.hann(frame_length * 2)
        
        n_speakers = len(np.unique(labels))
        
        for speaker in range(n_speakers):
            mask = (labels == speaker)
            audio_mask = np.repeat(mask, hop_length)
            if len(audio_mask) > len(audio):
                audio_mask = audio_mask[:len(audio)]
            elif len(audio_mask) < len(audio):
                audio_mask = np.pad(audio_mask, (0, len(audio) - len(audio_mask)))
            
            smoothed_audio = audio * audio_mask
            speaker_audio[f"speaker_{speaker}"] = smoothed_audio
        
        return speaker_audio
    
    def analyze_speakers(self, speaker_audio, sr):
        speaker_stats = {}
        
        for speaker_id, audio in speaker_audio.items():
            speaking_time = len(audio) / sr
            
            energy = np.abs(audio)
            is_speech = energy > np.mean(energy) * 0.1
            actual_speaking_time = np.sum(is_speech) / sr
            
            speech_energy = np.mean(energy[is_speech])
            
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mask = magnitudes > np.max(magnitudes) * 0.1
            pitch_mean = np.mean(pitches[pitch_mask]) if np.any(pitch_mask) else 0
            
            speaker_stats[speaker_id] = {
                'total_duration': speaking_time,
                'speech_duration': actual_speaking_time,
                'average_energy': speech_energy,
                'average_pitch': pitch_mean
            }
        
        return speaker_stats
    
    def process_audio(self, audio_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading audio...")
        audio, sr = self.load_audio(audio_path)
        
        print("Extracting spectral features...")
        features, hop_length = self.extract_features(audio, sr)
        
        n_speakers = self.estimate_number_of_speakers(features)
        
        print("Segmenting speakers...")
        labels = self.segment_speakers(features, n_speakers)
        
        print("Separating speaker audio...")
        speaker_audio = self.separate_speakers(audio, sr, labels, hop_length)
        
        print("Analyzing speaker characteristics...")
        speaker_stats = self.analyze_speakers(speaker_audio, sr)
        
        print("\nSaving separated audio files...")
        for speaker_id, audio_data in speaker_audio.items():
            output_path = os.path.join(output_dir, f"{speaker_id}.wav")
            sf.write(output_path, audio_data, sr)
            
            stats = speaker_stats[speaker_id]
            print(f"\n{speaker_id.upper()} Statistics:")
            print(f"Total Duration: {stats['total_duration']:.2f} seconds")
            print(f"Speech Duration: {stats['speech_duration']:.2f} seconds")
            print(f"Average Speech Energy: {stats['average_energy']:.5f}")
            print(f"Average Pitch: {stats['average_pitch']:.2f} Hz")
        
        return speaker_audio, speaker_stats

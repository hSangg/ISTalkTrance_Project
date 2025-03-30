import io

import librosa
import numpy as np

from modules.config import Config


class FeatureExtractor:
    @staticmethod
    def extract_mfcc_from_bytes(audio_bytes):
        try:
            audio, sr = librosa.load(
                io.BytesIO(audio_bytes), 
                sr=Config.SAMPLE_RATE
            )
            mfcc_features = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=Config.N_MFCC
            )
            
            mfcc_delta = librosa.feature.delta(mfcc_features)
            mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
            
            combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
            
            return combined_mfcc.T
        except Exception:
            raise

    @staticmethod
    def extract_features(audio_path, annotations):
        y, sr = librosa.load(audio_path, sr=None)
        speaker_data = {}
        for start, end, speaker in annotations:
            segment = y[int(start * sr): int(end * sr)]
            # mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T
            mfcc = FeatureExtractor.extract_mfcc(segment, sr)

            # Skip if MFCC extraction failed
            if mfcc is None:
                print(f"Skipping segment {start}-{end} for {speaker} due to MFCC extraction failure")
                continue

            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(mfcc)
        return speaker_data

    @staticmethod
    def extract_mfcc(segment, sr):
        """Extract MFCC features from a NumPy audio segment"""
        try:
            if len(segment) < sr * 0.025:
                segment = np.pad(segment, (0, int(sr * 0.025) - len(segment)))

            mfcc_features = librosa.feature.mfcc(
                y=segment,
                sr=sr,
                n_mfcc=Config.N_MFCC
            )
            mfcc_delta = librosa.feature.delta(mfcc_features)
            mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
            combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
            return combined_mfcc.T
        except Exception as e:
            print(f"Error extracting MFCC for segment of length {len(segment)}: {e}")
            return None
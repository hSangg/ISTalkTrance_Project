import librosa
import numpy as np
import io
from .config import Config

class FeatureExtractor:
    @staticmethod
    def extract_mfcc_from_bytes(audio_bytes):
        """Extract MFCC features from audio bytes"""
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
        except Exception as e:
            raise
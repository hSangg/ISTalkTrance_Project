import os
from typing import Dict, List

import joblib
import librosa
import numpy as np


class VoiceAuthenticator:
    def __init__(self, 
                 models_dir='models', 
                 test_dir='test', 
                 sample_rate=16000, 
                 n_mfcc=13):
        """
        Voice authentication and speaker identification
        
        Args:
            models_dir (str): Directory containing trained models
            test_dir (str): Directory with test audio files
            sample_rate (int): Audio sample rate for feature extraction
            n_mfcc (int): Number of MFCCs to extract
        """
        self.models_dir = models_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        # Load all models
        self.models = self.load_all_models()

    def load_all_models(self) -> Dict:
        """
        Load all trained models from models directory
        
        Returns:
            Dict of user models
        """
        models = {}
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_model.pkl'):
                user_id = filename.replace('_model.pkl', '')
                model_path = os.path.join(self.models_dir, filename)
                try:
                    model = joblib.load(model_path)
                    models[user_id] = model
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
        return models

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from an audio file
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=self.n_mfcc
            )
            delta1 = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            combined_features = np.vstack([mfcc, delta1, delta2]).T
            return combined_features
        
        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return np.array([])

    def predict_speaker(self, features: np.ndarray) -> Dict:
        """
        Predict the most likely speaker based on log likelihood
        
        Args:
            features (np.ndarray): Audio features to test
        
        Returns:
            Dict with prediction results
        """
        if len(features) == 0:
            return {"prediction": None, "scores": {}}
        
        # Calculate log likelihood for each model
        scores = {}
        for user_id, model in self.models.items():
            try:
                score = model.score(features)
                scores[user_id] = score
            except Exception as e:
                print(f"Scoring error for {user_id}: {e}")
        
        # Find the most likely speaker
        if scores:
            prediction = max(scores, key=scores.get)
            return {
                "prediction": prediction, 
                "scores": scores
            }
        
        return {"prediction": None, "scores": {}}

    def authenticate_test_files(self) -> List[Dict]:
        """
        Authenticate all .wav files in the test directory
        
        Returns:
            List of authentication results for each file
        """
        results = []
        
        # Find all .wav files in test directory
        wav_files = [
            f for f in os.listdir(self.test_dir) 
            if f.lower().endswith('.wav')
        ]
        
        for wav_file in wav_files:
            file_path = os.path.join(self.test_dir, wav_file)
            
            # Extract features
            features = self.extract_features(file_path)
            
            # Predict speaker
            prediction = self.predict_speaker(features)
            
            result = {
                "filename": wav_file,
                "prediction": prediction["prediction"],
                "scores": prediction["scores"]
            }
            results.append(result)
            
            # Print detailed results
            print(f"\nFile: {wav_file}")
            print(f"Predicted Speaker: {result['prediction']}")
            print("Likelihood Scores:")
            for user, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {user}: {score}")
        
        return results

# Usage example
if __name__ == "__main__":
    authenticator = VoiceAuthenticator()
    authentication_results = authenticator.authenticate_test_files()
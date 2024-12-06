import os
import numpy as np
from typing import Dict, List
import librosa
import joblib

class VoiceSimilarityChecker:
    def __init__(self, 
                 models_dir='models', 
                 sample_rate=16000, 
                 n_mfcc=13):
        """
        Voice similarity comparison tool
        
        Args:
            models_dir (str): Directory containing trained models
            sample_rate (int): Audio sample rate for feature extraction
            n_mfcc (int): Number of MFCCs to extract
        """
        self.models_dir = models_dir
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

    def compare_audio_files(self, file1_path: str, file2_path: str) -> Dict:
        """
        Compare two audio files to determine speaker similarity
        
        Args:
            file1_path (str): Path to first audio file
            file2_path (str): Path to second audio file
        
        Returns:
            Dict with comparison results
        """
        # Extract features for both files
        features1 = self.extract_features(file1_path)
        features2 = self.extract_features(file2_path)
        
        if len(features1) == 0 or len(features2) == 0:
            return {
                "similar": False, 
                "error": "Feature extraction failed"
            }
        
        # Compute log likelihoods
        results = []
        for user_id, model in self.models.items():
            try:
                # Score features of first file against model
                score1 = model.score(features1)
                
                # Score features of second file against same model
                score2 = model.score(features2)
                
                # Compute average score
                avg_score = (score1 + score2) / 2
                
                results.append({
                    "user_id": user_id,
                    "score": avg_score
                })
            except Exception as e:
                print(f"Scoring error for {user_id}: {e}")
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine similarity
        if results:
            best_match = results[0]
            # You can adjust this threshold based on your specific use case
            is_similar = best_match['score'] > -1000  # Example threshold
            
            return {
                "similar": is_similar,
                "best_match": best_match['user_id'],
                "scores": results
            }
        
        return {
            "similar": False,
            "error": "No valid comparisons"
        }

# Example usage
def main():
    # Initialize the comparison tool
    checker = VoiceSimilarityChecker()
    
    # Paths to your two audio files to compare
    file1 = 'test/file1.wav'
    file2 = 'test/file2.wav'
    
    # Compare the files
    comparison_result = checker.compare_audio_files(file1, file2)
    
    # Print results
    print("\nVoice Similarity Comparison:")
    print(f"Are files similar? {comparison_result.get('similar', False)}")
    if comparison_result.get('best_match'):
        print(f"Best match: {comparison_result['best_match']}")
    
    # Print detailed scores if available
    if comparison_result.get('scores'):
        print("\nDetailed Scores:")
        for result in comparison_result['scores']:
            print(f"{result['user_id']}: {result['score']}")

if __name__ == "__main__":
    main()
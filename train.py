import os
import numpy as np
import io
import librosa
from typing import Dict, List, Union

class EnhancedBatchTrainer:
    def __init__(self, 
                 train_dir='train_voice', 
                 sample_rate=16000, 
                 n_mfcc=13, 
                 hmm_components=5, 
                 hmm_iterations=100):
        """
        Enhanced batch trainer for voice authentication
        
        Args:
            train_dir (str): Directory containing user voice recordings
            sample_rate (int): Audio sample rate for feature extraction
            n_mfcc (int): Number of MFCCs to extract
            hmm_components (int): Number of HMM states
            hmm_iterations (int): Maximum HMM training iterations
        """
        self.train_dir = train_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hmm_components = hmm_components
        self.hmm_iterations = hmm_iterations
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)

    def parse_timestamp_script(self, script_path: str) -> List[Dict[str, Union[float, str]]]:
        """
        Parse timestamp script file into structured segments
        
        Args:
            script_path (str): Path to timestamp script file
        
        Returns:
            List of dictionaries with segment details
        """
        segments = []
        try:
            with open(script_path, 'r') as f:
                script_text = f.read().strip()
                time_labels = script_text.split()
                
            for i in range(0, len(time_labels), 3):
                start_time = self._parse_time(time_labels[i])
                end_time = self._parse_time(time_labels[i+1])
                label = time_labels[i+2]
                print(label)
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': label
                })
            
            return segments
        except Exception as e:
            print(f"Error parsing timestamp script: {e}")
            return []

    def _parse_time(self, time_str: str) -> float:
        """
        Convert timestamp to seconds
        
        Args:
            time_str (str): Timestamp in format M:SS or H:MM:SS
        
        Returns:
            float: Total seconds
        """
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0.0

    def extract_segmented_features(self, audio_path: str, segments: List[Dict]) -> np.ndarray:
        """
        Extract MFCC features for specific audio segments
        
        Args:
            audio_path (str): Path to audio file
            segments (List[Dict]): List of audio segments
        
        Returns:
            np.ndarray: Combined MFCC features
        """
        try:
            # Load entire audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Collect features for each segment
            segment_features = []
            for segment in segments:
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Extract features for this segment
                mfcc = librosa.feature.mfcc(
                    y=segment_audio, 
                    sr=sr, 
                    n_mfcc=self.n_mfcc
                )
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                
                combined_features = np.vstack([mfcc, delta1, delta2]).T
                segment_features.append(combined_features)
            
            # Combine all segment features
            return np.concatenate(segment_features)
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([])

    def train_user_model(self, user_dir: str) -> Dict:
        """
        Train a model for a specific user
        
        Args:
            user_dir (str): Directory containing user's audio and script
        
        Returns:
            Dict with training results
        """
        try:
            # Find raw audio and timestamp script
            raw_audio = os.path.join(user_dir, 'raw.wav')
            script_file = [f for f in os.listdir(user_dir) if f.endswith('.txt')][0]
            script_path = os.path.join(user_dir, script_file)
            
            # Parse timestamp script
            segments = self.parse_timestamp_script(script_path)
            
            # Extract features
            features = self.extract_segmented_features(raw_audio, segments)
            
            if len(features) == 0:
                return {
                    'success': False,
                    'user_id': os.path.basename(user_dir),
                    'error': 'No features extracted'
                }
            
            # Train HMM model
            from hmmlearn import hmm
            model = hmm.GaussianHMM(
                n_components=self.hmm_components, 
                covariance_type='diag', 
                n_iter=self.hmm_iterations
            )
            model.fit(features)
            
            # Save model
            user_id = os.path.basename(user_dir)
            model_path = os.path.join('models', f'{user_id}_model.pkl')
            import joblib
            joblib.dump(model, model_path)
            
            return {
                'success': True,
                'user_id': user_id,
                'features_shape': features.shape
            }
        
        except Exception as e:
            return {
                'success': False,
                'user_id': os.path.basename(user_dir),
                'error': str(e)
            }

    def train_all_users(self) -> Dict:
        """
        Train models for all users in train_voice directory
        
        Returns:
            Dict with overall training results
        """
        results = []
        total_users = 0
        successful_users = 0
        
        for user_dir in [os.path.join(self.train_dir, d) for d in os.listdir(self.train_dir) 
                         if os.path.isdir(os.path.join(self.train_dir, d))]:
            total_users += 1
            result = self.train_user_model(user_dir)
            results.append(result)
            
            if result['success']:
                successful_users += 1
                print(f"Successfully trained model for {result['user_id']}")
            else:
                print(f"Failed to train model for {result['user_id']}: {result.get('error', 'Unknown error')}")
        
        return {
            'total_users': total_users,
            'successful_trainings': successful_users,
            'failed_trainings': total_users - successful_users,
            'details': results
        }

# Usage example
if __name__ == "__main__":
    trainer = EnhancedBatchTrainer()
    training_results = trainer.train_all_users()
    print(training_results)
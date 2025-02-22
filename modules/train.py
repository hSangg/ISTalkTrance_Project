import os
import numpy as np
import librosa
from typing import Dict, List, Union
from hmmlearn import hmm
import joblib
import soundfile as sf
from sklearn.model_selection import KFold
import numpy as np
import optuna

class EnhancedBatchTrainer:
    def __init__(self, 
                 train_dir='train_voice', 
                 sample_rate=16000, 
                 n_mfcc=13, 
                 hmm_components=5, 
                 hmm_iterations=100,
                 n_trials: int = 5):
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
        self.n_trials = n_trials

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
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': label
                })
            audio_path = os.path.join('train_voice', 'user123', 'raw.wav')
            print(f"Type of audio_path: {type(audio_path)}")

            output_dir='20_percent_test'
            if segments:
                self.extract_and_export_20_percent(audio_path, segments, output_dir)
            else:
                print("No segments to extract.")
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

    def extract_segmented_features(self, audio_path: str, segments: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features for each label in the segments.

        Args:
            audio_path (str): Path to the audio file.
            segments (List[Dict]): List of audio segments.

        Returns:
            Dict[str, np.ndarray]: Features grouped by label.
        """
        try:
            # Load entire audio
            print(f"Loading audio from: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Minimum number of samples required for MFCC extraction
            min_samples = 2048  # Default `n_fft` for STFT in librosa

            # Collect features for each label
            label_features = {}
            for segment in segments:
                label = segment['label']
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)

                # Extract the segment
                segment_audio = audio[start_sample:end_sample]

                # Check if the segment is long enough
                if len(segment_audio) < min_samples:
                    print(f"Skipping segment {label}: too short ({len(segment_audio)} samples).")
                    continue

                # Extract features for this segment
                mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=self.n_mfcc)
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)

                combined_features = np.vstack([mfcc, delta1, delta2]).T

                # Store features by label
                if label not in label_features:
                    label_features[label] = combined_features
                else:
                    label_features[label] = np.vstack([label_features[label], combined_features])

            return label_features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}

    def objective(self, trial, features: np.ndarray, n_splits: int = 5):
        """
        Optuna objective function for optimizing HMM parameters
        """
        # Define hyperparameter search space
        n_components = trial.suggest_int('n_components', 2, 10)
        covariance_type = trial.suggest_categorical('covariance_type', 
                                                  ['diag', 'spherical', 'tied', 'full'])
        n_iter = trial.suggest_int('n_iter', 50, 200)
        
        # Cross validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(features):
            train_features = features[train_idx]
            test_features = features[test_idx]
            
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=42
            )
            
            try:
                model.fit(train_features)
                score = model.score(test_features)
                scores.append(score)
            except Exception as e:
                # Return a very low score if model fitting fails
                return float('-inf')
        
        return np.mean(scores)

    def train_user_model(self, user_dir: str, n_splits: int = 5) -> Dict:
        """
        Train models for a specific user using Optuna for hyperparameter optimization
        and k-fold cross validation for evaluation.

        Args:
            user_dir (str): Directory containing the user's audio and script
            n_splits (int): Number of folds for cross-validation (default: 5)

        Returns:
            Dict with training results including cross-validation scores and best parameters
        """
        try:
            raw_audio = os.path.join(user_dir, 'raw.wav')
            script_file = [f for f in os.listdir(user_dir) if f.endswith('.txt')][0]
            script_path = os.path.join(user_dir, script_file)
            segments = self.parse_timestamp_script(script_path)

            results = {}
            
            # Process each label separately
            label_features = self.extract_segmented_features(raw_audio, segments)
            
            for label, features in label_features.items():
                if len(features) == 0:
                    results[label] = {
                        'success': False,
                        'error': f'No features extracted for label {label}'
                    }
                    continue

                # Create a new study for this label
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self.objective(trial, features, n_splits), 
                             n_trials=self.n_trials)
                
                # Get best parameters
                best_params = study.best_params
                
                # Train final model with best parameters on all data
                final_model = hmm.GaussianHMM(
                    n_components=best_params['n_components'],
                    covariance_type=best_params['covariance_type'],
                    n_iter=best_params['n_iter'],
                    random_state=42
                )
                final_model.fit(features)
                
                # Save the final model
                model_path = os.path.join('models', f'{label}_model.pkl')
                joblib.dump(final_model, model_path)
                
                # Evaluate with cross-validation using best parameters
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = []
                
                for train_idx, test_idx in kf.split(features):
                    train_features = features[train_idx]
                    test_features = features[test_idx]
                    
                    model = hmm.GaussianHMM(
                        n_components=best_params['n_components'],
                        covariance_type=best_params['covariance_type'],
                        n_iter=best_params['n_iter'],
                        random_state=42
                    )
                    model.fit(train_features)
                    score = model.score(test_features)
                    cv_scores.append(score)
                
                # Store results for this label
                results[label] = {
                    'success': True,
                    'model_path': model_path,
                    'best_parameters': best_params,
                    'best_trial_score': study.best_value,
                    'cv_scores': cv_scores,
                    'mean_cv_score': np.mean(cv_scores),
                    'std_cv_score': np.std(cv_scores)
                }

            return {
                'success': True,
                'user_id': os.path.basename(user_dir),
                'results': results
            }
        except Exception as e:
            return {
                'success': False,
                'user_id': os.path.basename(user_dir),
                'error': str(e)
            }

    
    def extract_and_export_20_percent(self, audio_path: str, segments: List, output_dir: str):
        """
        Use the first 80% of the segments for training and export the last 20% for testing.

        Args:
            audio_path (str): Path to the full audio file
            segments (List): List of segments to process
            output_dir (str): Directory to save the exported audio files
        """
        print("Starting extract_and_export_20_percent function")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return

        num_segments = len(segments)
        test_size = int(num_segments * 0.2) 
        train_segments = segments[:num_segments - test_size] 
        test_segments = segments[num_segments - test_size:] 

        print(f"Number of segments: {num_segments}")
        print(f"Training segments: {len(train_segments)}, Testing segments: {len(test_segments)}")

        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"Audio loaded from {audio_path}, sample rate: {sr}")

            for i, segment in enumerate(test_segments):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]

                output_audio_path = os.path.join(output_dir, f"test_segment_{i+1}.wav")
                sf.write(output_audio_path, segment_audio, sr)
                print(f"Exported test segment {i+1} to {output_audio_path}")

            print("Test segments exported successfully.")
        except Exception as e:
            print(f"Error exporting test segments: {e}")

        return train_segments


if __name__ == "__main__":
    trainer = EnhancedBatchTrainer()
    training_results = trainer.train_all_users()
    print(training_results)

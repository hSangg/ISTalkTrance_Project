import os
from typing import Dict, List

import joblib
import librosa
import numpy as np
import optuna
import soundfile as sf
from hmmlearn import hmm
from sklearn.model_selection import KFold

from modules.config import Config
from modules.feature_extractor import FeatureExtractor
from modules.model_manager import ModelManager
from modules.utils import Utils


class Trainner:
    def __init__(self, 
                 train_dir='train_voice', 
                 sample_rate=16000, 
                 n_mfcc=13, 
                 hmm_components=5, 
                 hmm_iterations=100,
                 n_trials: int = 5):

        self.train_dir = train_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hmm_components = hmm_components
        self.hmm_iterations = hmm_iterations
        self.n_trials = n_trials

        os.makedirs('models', exist_ok=True)

    @staticmethod
    def train_hmm_model_all():
        speaker_data = {}
        subfolders = [f.path for f in os.scandir(Config.TRAIN_VOICE) if f.is_dir()]

        for subfolder in subfolders:
            print("✨ start extract feature for sub-folder ✨ \t \t", subfolder)
            annotation_file = os.path.join(subfolder, "script.txt")
            audio_file = os.path.join(subfolder, "raw.wav")

            if not os.path.exists(annotation_file) or not os.path.exists(audio_file):
                continue

            annotations = Utils.load_annotations(annotation_file)
            folder_speaker_data = FeatureExtractor.extract_append_features(audio_file, annotations)

            for speaker, data in folder_speaker_data.items():
                if speaker not in speaker_data:
                    speaker_data[speaker] = []
                speaker_data[speaker].extend(data)

        for speaker, data in speaker_data.items():
            print("✨ start train for: ", speaker, " ✨")
            ModelManager.train_hmm_model(speaker, data)

    def extract_segmented_features(self, audio_path: str, segments: List[Dict]) -> Dict[str, np.ndarray]:
        try:
            print(f"Loading audio from: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            min_samples = 2048

            label_features = {}
            for segment in segments:
                label = segment['label']
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)

                segment_audio = audio[start_sample:end_sample]

                if len(segment_audio) < min_samples:
                    print(f"Skipping segment {label}: too short ({len(segment_audio)} samples).")
                    continue

                mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=self.n_mfcc)
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)

                combined_features = np.vstack([mfcc, delta1, delta2]).T

                if label not in label_features:
                    label_features[label] = combined_features
                else:
                    label_features[label] = np.vstack([label_features[label], combined_features])

            return label_features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}

    def objective(self, trial, features: np.ndarray, n_splits: int = 5):
        n_components = trial.suggest_int('n_components', 2, 10)
        covariance_type = trial.suggest_categorical('covariance_type', 
                                                  ['diag', 'spherical', 'tied', 'full'])
        n_iter = trial.suggest_int('n_iter', 50, 200)
        
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
            except Exception:
                return float('-inf')
        
        return np.mean(scores)

    def train_user_model(self, user_dir: str, n_splits: int = 5) -> Dict:
        try:
            raw_audio = os.path.join(user_dir, 'raw.wav')
            script_file = [f for f in os.listdir(user_dir) if f.endswith('.txt')][0]
            script_path = os.path.join(user_dir, script_file)
            segments = Utils.parse_timestamp_script(self, script_path)

            results = {}
            
            label_features = self.extract_segmented_features(raw_audio, segments)
            
            for label, features in label_features.items():
                if len(features) == 0:
                    results[label] = {
                        'success': False,
                        'error': f'No features extracted for label {label}'
                    }
                    continue

                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self.objective(trial, features, n_splits), 
                             n_trials=self.n_trials)
                
                best_params = study.best_params
                
                final_model = hmm.GaussianHMM(
                    n_components=best_params['n_components'],
                    covariance_type=best_params['covariance_type'],
                    n_iter=best_params['n_iter'],
                    random_state=42
                )
                final_model.fit(features)
                
                model_path = os.path.join('models', f'{label}_model.pkl')
                joblib.dump(final_model, model_path)
                
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


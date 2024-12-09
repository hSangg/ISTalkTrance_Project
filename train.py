import os
import numpy as np
import librosa
from typing import Dict, List, Union
from hmmlearn import hmm
import joblib
import soundfile as sf


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
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': label
                })
            # user_dir = './train_voice/user123/'
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


    def train_user_model(self, user_dir: str) -> Dict:
        """
        Train separate models for each label in the user's data

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
            label_features = self.extract_segmented_features(raw_audio, segments)
            
            results = {}
            for label, features in label_features.items():
                if len(features) == 0:
                    results[label] = {
                        'success': False,
                        'error': f'No features extracted for label {label}'
                    }
                    continue
                
                # Train HMM model
                model = hmm.GaussianHMM(
                    n_components=self.hmm_components, 
                    covariance_type='diag', 
                    n_iter=self.hmm_iterations
                )
                model.fit(features)
                
                # Save model
                model_path = os.path.join('models', f'{label}_model.pkl')
                joblib.dump(model, model_path)
                
                results[label] = {'success': True, 'model_path': model_path}
            
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

    def train_user_model(self, user_dir: str) -> Dict:
        """
        Train models for a specific user using the first 80% of segments for training
        and the last 20% for testing.

        Args:
            user_dir (str): Directory containing the user's audio and script

        Returns:
            Dict with training results
        """
        try:
            # Locate raw audio and script file
            raw_audio = os.path.join(user_dir, 'raw.wav')
            script_file = [f for f in os.listdir(user_dir) if f.endswith('.txt')][0]
            script_path = os.path.join(user_dir, script_file)

            # Parse timestamp script
            segments = self.parse_timestamp_script(script_path)

            # Export the last 20% for testing and get the first 80% for training
            train_segments = self.extract_and_export_20_percent(raw_audio, segments, output_dir="20_percent_test")

            # Extract features for training data
            label_features = self.extract_segmented_features(raw_audio, train_segments)

            results = {}
            for label, features in label_features.items():
                if len(features) == 0:
                    results[label] = {
                        'success': False,
                        'error': f'No features extracted for label {label}'
                    }
                    continue

                # Train HMM model
                model = hmm.GaussianHMM(
                    n_components=self.hmm_components,
                    covariance_type='diag',
                    n_iter=self.hmm_iterations
                )
                model.fit(features)

                # Save model
                model_path = os.path.join('models', f'{label}_model.pkl')
                joblib.dump(model, model_path)

                results[label] = {'success': True, 'model_path': model_path}

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

        # Split segments into training (80%) and testing (20%)
        num_segments = len(segments)
        test_size = int(num_segments * 0.2)  # Last 20% for testing
        train_segments = segments[:num_segments - test_size]  # First 80% for training
        test_segments = segments[num_segments - test_size:]  # Last 20% for testing

        print(f"Number of segments: {num_segments}")
        print(f"Training segments: {len(train_segments)}, Testing segments: {len(test_segments)}")

        # Export test segments (last 20%) to output directory
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            print(f"Audio loaded from {audio_path}, sample rate: {sr}")

            for i, segment in enumerate(test_segments):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]

                # Save the segment as a new file
                output_audio_path = os.path.join(output_dir, f"test_segment_{i+1}.wav")
                sf.write(output_audio_path, segment_audio, sr)
                print(f"Exported test segment {i+1} to {output_audio_path}")

            print("Test segments exported successfully.")
        except Exception as e:
            print(f"Error exporting test segments: {e}")

        # Return the training segments for further processing
        return train_segments


# Usage example
if __name__ == "__main__":
    trainer = EnhancedBatchTrainer()
    training_results = trainer.train_all_users()
    print(training_results)

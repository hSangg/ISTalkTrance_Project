import librosa
import numpy as np
import os
import joblib
from hmmlearn import hmm
import soundfile as sf
from typing import List, Tuple, Dict, Any


def time_to_seconds(time_str: str) -> float:
    """Convert time string to seconds. Handles h:mm:ss format.
    
    Args:
        time_str: Time string in h:mm:ss format (e.g., '0:09:31')
        
    Returns:
        Float value representing total seconds
    """
    try:
        # Split time string into parts
        parts = time_str.strip().split(':')
        
        if len(parts) != 3:
            raise ValueError(f"Expected h:mm:ss format with 3 parts, got {len(parts)} parts")
            
        # Convert parts to integers/floats
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        
        # Validate ranges
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"Invalid minutes/seconds value. Minutes: {minutes}, Seconds: {seconds}")
            
        # Calculate total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
        
    except ValueError as e:
        raise ValueError(f"Invalid time format in '{time_str}': {str(e)}")


def extract_mfcc_librosa(audio_segment: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract MFCC features with deltas and delta-deltas."""
    try:
        mfcc = librosa.feature.mfcc(
            y=audio_segment,
            sr=sr,
            n_mfcc=13,
            n_fft=1200,
            hop_length=int(sr * 0.01),
            win_length=int(sr * 0.025),
            dtype=np.float32
        )
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        # Combine MFCC, delta, and delta-delta
        features = np.vstack([mfcc, delta, delta_delta]).T  # (Frames, Features)
        return features
    except Exception as e:
        print(f"Error in MFCC extraction: {str(e)}")
        return np.empty((0, 39))  # Return consistent empty array


def load_models(models_dir: str) -> Dict[str, hmm.GaussianHMM]:
    """Load HMM models from the specified directory."""
    models = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.pkl'):
            speaker_id = model_file.split('_')[0]
            try:
                model_path = os.path.join(models_dir, model_file)
                models[speaker_id] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model {model_file}: {str(e)}")
    return models


def process_audio_segment(
    y: np.ndarray, sr: int, start: float, end: float
) -> Tuple[np.ndarray, bool]:
    """Extract a segment from audio and validate its length."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    if end_sample > len(y):
        print(f"Segment end {end_sample} exceeds signal length {len(y)}.")
        return np.array([]), False

    segment_audio = y[start_sample:end_sample]
    if len(segment_audio) < sr * 0.1:  # Skip segments shorter than 100ms
        print("Segment too short to process.")
        return np.array([]), False

    return segment_audio, True

def calculate_speaker_metrics(
    confusion_matrix: np.ndarray, speaker_map: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    """Calculate precision, recall, and F1-score for each speaker."""
    metrics = {}
    for speaker, idx in speaker_map.items():
        true_positives = confusion_matrix[idx, idx]
        total_actual = confusion_matrix[idx].sum()
        total_predicted = confusion_matrix[:, idx].sum()

        precision = true_positives / total_predicted if total_predicted else 0.0
        recall = true_positives / total_actual if total_actual else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[speaker] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }
    return metrics

def evaluate_speaker_recognition(
    train_voice_dir: str, models_dir: str
) -> Dict[str, Any]:
    """Evaluate speaker recognition system."""
    models = load_models(models_dir)
    if not models:
        raise ValueError("No valid models found in the specified directory.")

    speaker_map = {speaker: idx for idx, speaker in enumerate(models.keys())}
    confusion_matrix = np.zeros((len(models), len(models)))
    total_segments = 0
    correct_predictions = 0
    skipped_segments = 0
    errors = []

    for folder in os.listdir(train_voice_dir):
        folder_path = os.path.join(train_voice_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not wav_files:
            print(f"No WAV files found in folder: {folder}")
            continue

        wav_file = os.path.join(folder_path, wav_files[0])
        script_file = os.path.join(folder_path, 'script.txt')

        try:
            y, sr = sf.read(wav_file)
            if y.ndim > 1:  # Ensure mono audio
                y = librosa.to_mono(y.T)
                
            with open(script_file, 'r', encoding='utf-8') as f:
                segments = []
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue
                            
                        start = time_to_seconds(parts[0])
                        end = time_to_seconds(parts[1])
                        speaker = parts[2]
                        
                        # Basic validation
                        if end <= start:
                            errors.append(f"Line {line_num} in {folder}: End time {end} not after start time {start}")
                            continue
                            
                        segments.append((start, end, speaker))
                        
                    except ValueError as e:
                        errors.append(f"Line {line_num} in {folder}: {str(e)}")
                        continue

            for start, end, true_speaker in segments:
                if true_speaker not in speaker_map:
                    errors.append(f"Unknown speaker {true_speaker} in folder {folder}")
                    skipped_segments += 1
                    continue

                segment_audio, valid = process_audio_segment(y, sr, start, end)
                if not valid:
                    skipped_segments += 1
                    continue

                segment_features = extract_mfcc_librosa(segment_audio, sr)
                if segment_features.size == 0:
                    skipped_segments += 1
                    continue

                scores = {
                    speaker: model.score(segment_features)
                    for speaker, model in models.items()
                }
                predicted_speaker = max(scores, key=scores.get)
                true_idx = speaker_map[true_speaker]
                pred_idx = speaker_map[predicted_speaker]

                total_segments += 1
                correct_predictions += int(predicted_speaker == true_speaker)
                confusion_matrix[true_idx, pred_idx] += 1

        except Exception as e:
            errors.append(f"Error processing folder {folder}: {str(e)}")
            continue

    accuracy = correct_predictions / total_segments if total_segments else 0.0
    speaker_metrics = calculate_speaker_metrics(confusion_matrix, speaker_map)

    return {
        'overall_accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'speaker_metrics': speaker_metrics,
        'total_segments': total_segments,
        'skipped_segments': skipped_segments,
        'errors': errors
    }

def print_evaluation_results(results: Dict[str, Any]) -> None:
    """Print formatted evaluation results."""
    print("\nEvaluation Results:")
    print(f"Total segments processed: {results['total_segments']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print("\nPer-speaker metrics:")
    for speaker, metrics in results['speaker_metrics'].items():
        print(f"\nSpeaker {speaker}:")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1-score: {metrics['f1_score']:.2%}")


if __name__ == "__main__":
    train_voice_dir = "train_voice"
    models_dir = "models"
    results = evaluate_speaker_recognition(train_voice_dir, models_dir)
    print_evaluation_results(results)

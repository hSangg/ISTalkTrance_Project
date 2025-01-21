import numpy as np
import os
import librosa
import soundfile as sf
import io
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict, Any
from modules.authenticate import VoiceAuthenticator


authenticator = VoiceAuthenticator()

def time_to_seconds(time_str: str) -> float:
    """Convert time string to seconds. Handles h:mm:ss format."""
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 3:
            raise ValueError(f"Expected h:mm:ss format with 3 parts, got {len(parts)} parts")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError as e:
        raise ValueError(f"Invalid time format in '{time_str}': {str(e)}")

def process_audio_segment(y: np.ndarray, sr: int, start: float, end: float) -> Tuple[bytes, bool]:
    """Extract a segment from audio and return it as a bytes-like object."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    if end_sample > len(y):
        print(f"Segment end {end_sample} exceeds signal length {len(y)}.")
        return b"", False
    segment_audio = y[start_sample:end_sample]
    if len(segment_audio) < sr * 0.1:  # Segment must be long enough
        print("Segment too short to process.")
        return b"", False
    byte_io = io.BytesIO()
    sf.write(byte_io, segment_audio, sr, format='WAV')
    audio_bytes = byte_io.getvalue()
    return audio_bytes, True

def test_directory(directory_path: str):
    """Process the entire directory and subdirectories for authentication testing."""
    results = defaultdict(list)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    total_segments = 0
    correct_predictions = 0

    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue

        wav_file = os.path.join(folder_path, 'raw.wav')
        script_file = os.path.join(folder_path, 'script.txt')

        if not os.path.exists(wav_file) or not os.path.exists(script_file):
            print(f"Skipping folder {folder} - missing WAV or script.txt")
            continue

        try:
            y, sr = sf.read(wav_file)
            if y.ndim > 1:
                y = librosa.to_mono(y.T)  # Convert stereo to mono

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
                        if end <= start:
                            print(f"Invalid time range in line {line_num} of {folder}: {line}")
                            continue
                        segments.append((start, end, speaker))
                    except ValueError as e:
                        print(f"Error processing line {line_num} in {folder}: {str(e)}")

            for start, end, true_speaker in segments:
                # Process audio segment
                audio_bytes, valid = process_audio_segment(y, sr, start, end)
                if not valid:
                    continue
                
                # Authenticate the user based on audio bytes
                result = authenticator.authenticate(audio_bytes)
                if result["success"]:
                    predicted_speaker = result['best_user']
                    confusion_matrix[true_speaker][predicted_speaker] += 1
                    if predicted_speaker == true_speaker:
                        correct_predictions += 1
                total_segments += 1
                results[true_speaker].append(result)

        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")

    # Calculate Precision, Recall, and F1-Score for each speaker
    precision_recall_f1 = {}
    for speaker in confusion_matrix:
        true_positives = confusion_matrix[speaker][speaker]
        false_positives = sum(confusion_matrix[speaker].values()) - true_positives
        false_negatives = sum(confusion_matrix[other_speaker][speaker] for other_speaker in confusion_matrix if other_speaker != speaker)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        
        precision_recall_f1[speaker] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }

    accuracy = correct_predictions / total_segments if total_segments else 0.0
    return {
        'accuracy': accuracy,
        'precision_recall_f1': precision_recall_f1,
        'confusion_matrix': confusion_matrix,
        'results': results
    }

if __name__ == "__main__":
    train_voice_dir = "train_voice"  # The root directory containing all subfolders
    results = test_directory(train_voice_dir)

    # Optionally print or save the results
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    print("\nPer-speaker metrics:")
    for speaker, metrics in results['precision_recall_f1'].items():
        print(f"\nSpeaker {speaker}:")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall: {metrics['recall']:.2%}")
        print(f"F1-score: {metrics['f1_score']:.2%}")

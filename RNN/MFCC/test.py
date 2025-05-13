import glob
import os

import librosa
import numpy as np
from tensorflow.keras.models import load_model


def extract_mfcc_segment(audio, sample_rate, start_time, end_time, n_mfcc=13, fixed_length=100):
    """Extract MFCC features from an audio segment."""
    try:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        if start_sample >= len(audio):
            return None

        end_sample = min(end_sample, len(audio))
        segment_audio = audio[start_sample:end_sample]

        mfccs = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs = mfccs.T

        if mfccs.shape[0] < fixed_length:
            pad_width = fixed_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:fixed_length, :]

        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None


def parse_script(script_path):
    """Parse script.txt file to get time segments."""
    segments = []
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('start'):
                parts = line.split()
                if len(parts) >= 2:
                    start_time = parts[0]
                    end_time = parts[1]

                    start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
                    end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))

                    segments.append({
                        'start': start_seconds,
                        'end': end_seconds,
                        'start_fmt': start_time,
                        'end_fmt': end_time
                    })
    return segments


def predict_speakers(audio_path, segments, model, speakers, fixed_length=100):
    """Predict speakers for each segment in the audio file."""
    results = []

    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)

        for segment in segments:
            mfccs = extract_mfcc_segment(
                audio, sample_rate, segment['start'], segment['end'],
                fixed_length=fixed_length
            )

            if mfccs is None:
                print(f"Warning: Could not extract features for segment {segment['start_fmt']}-{segment['end_fmt']}")
                predicted_speaker = "unknown"
            else:
                X = np.expand_dims(mfccs, axis=0)

                prediction = model.predict(X, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                predicted_speaker = speakers[predicted_class]
                confidence = prediction[0][predicted_class]

                print(
                    f"Segment {segment['start_fmt']}-{segment['end_fmt']}: Predicted {predicted_speaker} (confidence: {confidence:.4f})")

            results.append({
                'start': segment['start_fmt'],
                'end': segment['end_fmt'],
                'speaker': predicted_speaker
            })

        return results
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []


def write_predictions(results, output_path):
    """Write prediction results to script_predicted.txt."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("start end speaker\n")
        for result in results:
            f.write(f"{result['start']} {result['end']} {result['speaker']}\n")

    print(f"Predictions written to {output_path}")


def process_test_folders(test_root_folder, model_path, speakers_path, fixed_length=100):
    print(f"Loading model from {model_path}")
    model = load_model(model_path)

    print(f"Loading speakers from {speakers_path}")
    speakers = np.load(speakers_path, allow_pickle=True)
    print(f"Loaded {len(speakers)} speakers: {speakers}")

    subfolders = [f.path for f in os.scandir(test_root_folder) if f.is_dir()]
    print(f"Found {len(subfolders)} test subfolders")

    for subfolder in subfolders:
        wav_files = glob.glob(os.path.join(subfolder, "*.wav")) + glob.glob(os.path.join(subfolder, "*.WAV"))
        script_files = glob.glob(os.path.join(subfolder, "script.txt"))

        if wav_files and script_files:
            print(f"\nProcessing folder: {subfolder}")
            wav_file = wav_files[0]
            script_file = script_files[0]

            segments = parse_script(script_file)
            print(f"Found {len(segments)} segments in script")

            results = predict_speakers(wav_file, segments, model, speakers, fixed_length)

            if results:
                output_path = os.path.join(subfolder, "script_predicted.txt")
                write_predictions(results, output_path)


if __name__ == "__main__":
    TEST_ROOT_FOLDER = "test_voice"
    MODEL_PATH = "rnn_mfcc_models/speaker_recognition_rnn.keras"
    SPEAKERS_PATH = "rnn_mfcc_models/speakers.npy"

    FIXED_LENGTH = 100

    process_test_folders(TEST_ROOT_FOLDER, MODEL_PATH, SPEAKERS_PATH, FIXED_LENGTH)

    print("\nPrediction completed!")
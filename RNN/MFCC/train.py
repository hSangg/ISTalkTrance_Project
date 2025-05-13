import glob
import os

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.utils import to_categorical


def extract_mfcc(file_path, n_mfcc=13, fixed_length=None):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        mfccs = mfccs.T

        if fixed_length is not None:
            if mfccs.shape[0] < fixed_length:
                pad_width = fixed_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfccs = mfccs[:fixed_length, :]

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def parse_script(script_path):
    segments = []
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('start'):
                parts = line.split()
                if len(parts) >= 3:
                    start_time = parts[0]
                    end_time = parts[1]
                    speaker = parts[2]

                    start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
                    end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))

                    segments.append({
                        'start': start_seconds,
                        'end': end_seconds,
                        'speaker': speaker
                    })
    return segments


def segment_audio_and_extract_features(audio_path, segments, fixed_length=100):
    features = []
    labels = []

    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)

        for segment in segments:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)

            if start_sample >= end_sample or start_sample >= len(audio):
                continue

            end_sample = min(end_sample, len(audio))
            segment_audio = audio[start_sample:end_sample]

            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=13)
            mfccs = mfccs.T

            if mfccs.shape[0] < fixed_length:
                pad_width = fixed_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            else:
                mfccs = mfccs[:fixed_length, :]

            features.append(mfccs)
            labels.append(segment['speaker'])

        return features, labels
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return [], []


def collect_data(root_folder, fixed_length=100):
    all_features = []
    all_labels = []
    all_speakers = set()

    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    for subfolder in subfolders:
        wav_files = glob.glob(os.path.join(subfolder, "*.wav")) + glob.glob(os.path.join(subfolder, "*.WAV"))
        script_files = glob.glob(os.path.join(subfolder, "script.txt"))

        if wav_files and script_files:
            print(f"Processing folder: {subfolder}")
            wav_file = wav_files[0]
            script_file = script_files[0]

            segments = parse_script(script_file)

            features, labels = segment_audio_and_extract_features(wav_file, segments, fixed_length)

            if features and labels:
                all_features.extend(features)
                all_labels.extend(labels)
                all_speakers.update(set(labels))

    print(f"Found {len(all_speakers)} unique speakers: {all_speakers}")
    return np.array(all_features), np.array(all_labels), list(all_speakers)


def train_rnn_model(features, labels, speakers):
    label_encoder = LabelEncoder()
    label_encoder.fit(speakers)
    encoded_labels = label_encoder.transform(labels)

    n_classes = len(speakers)
    one_hot_labels = to_categorical(encoded_labels, num_classes=n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        features, one_hot_labels, test_size=0.2, random_state=42
    )

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(features.shape[1], features.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    return model, label_encoder, history

def export_model(model, label_encoder, speakers, output_dir="rnn_mfcc_models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "speaker_recognition_rnn.keras")
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    np.save(os.path.join(output_dir, "speakers.npy"), label_encoder.classes_)

    with open(os.path.join(output_dir, "speaker_mapping.txt"), "w") as f:
        for i, speaker in enumerate(label_encoder.classes_):
            f.write(f"{i}: {speaker}\n")

    print(f"Speaker mapping saved to {output_dir}/speaker_mapping.txt")


def predict_speaker(model, label_encoder, audio_path, fixed_length=100):
    mfccs = extract_mfcc(audio_path, fixed_length=fixed_length)

    if mfccs is None:
        return None

    X = np.expand_dims(mfccs, axis=0)

    prediction = model.predict(X)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_speaker = label_encoder.classes_[predicted_class]
    confidence = prediction[0][predicted_class]

    return {
        'speaker': predicted_speaker,
        'confidence': float(confidence),
        'all_probabilities': {speaker: float(prob) for speaker, prob in zip(label_encoder.classes_, prediction[0])}
    }


if __name__ == "__main__":
    ROOT_FOLDER = "train_voice"

    FIXED_LENGTH = 100

    print("Collecting data from all subfolders...")
    features, labels, speakers = collect_data(ROOT_FOLDER, fixed_length=FIXED_LENGTH)

    if len(features) == 0:
        print("No features extracted. Please check your data folder structure.")
        exit(1)

    print(f"Collected {len(features)} segments from {len(speakers)} speakers")

    print("Training RNN model...")
    model, label_encoder, history = train_rnn_model(features, labels, speakers)

    print("Exporting model...")
    export_model(model, label_encoder, speakers)

    print("Done!")


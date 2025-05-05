import json
import os
import pickle
import wave
from datetime import timedelta
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pyannote.audio import Pipeline
from transformers import (Pipeline, pipeline)
from werkzeug.datastructures import FileStorage

from modules.authenticate import VoiceAuthenticator
from modules.batch_trainer import BatchTrainer
from modules.config import Config
from modules.feature_extractor import FeatureExtractor
from modules.model_manager import ModelManager
from modules.trainner import Trainner
from modules.utils import Utils

COMPLETED = "Completed"
ERROR = "error"
OUTPUT = "output"
MESSAGE = "message"
MODEL_EXTENSION = ".pkl"
HHMMSS_FORMAT = "%H:%M:%S"
NO_FILES_PROVIDED = "No files provided"
SCRIPT = "script"
AUDIO = "audio"

app = Flask(__name__)
CORS(app)
Config.setup()

authenticator = VoiceAuthenticator()
batch_trainer = BatchTrainer()
hf_token_full_access = os.getenv("HF_TOKEN_FULL_ACCESS")
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", token=hf_token_full_access)

@app.route("/diarization", methods=["POST"])
def diarization():
    if AUDIO not in request.files:
        return jsonify({ERROR: NO_FILES_PROVIDED}), 400

    audio_file = request.files[AUDIO]
    audio_bytes = audio_file.read()

    audio_io = BytesIO(audio_bytes)

    with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
        framerate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()

    full_audio_io = BytesIO(audio_bytes)
    full_audio, sr = librosa.load(full_audio_io, sr=None, mono=True)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token_full_access)

    print("prepare pipeline...")
    diarization = pipeline(audio_io)

    results = []
    grouped_turns = []
    current_group = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if not current_group:
            current_group = [(turn, speaker)]
        else:
            _, last_speaker = current_group[-1]
            if speaker == last_speaker:
                current_group.append((turn, speaker))
            else:
                grouped_turns.append(current_group)
                current_group = [(turn, speaker)]

    if current_group:
        grouped_turns.append(current_group)

    for group in grouped_turns:
        start_time_sec = group[0][0].start
        end_time_sec = group[-1][0].end

        start_time = str(timedelta(seconds=int(start_time_sec)))
        end_time = str(timedelta(seconds=int(end_time_sec)))

        if start_time == end_time:
            continue

        start_sample = int(start_time_sec * sr)
        end_sample = int(end_time_sec * sr)

        audio_segment = full_audio[start_sample:end_sample]

        segment_wav = BytesIO()
        sf.write(segment_wav, audio_segment, sr, format='WAV')
        segment_wav.seek(0)

        print(f"🏃‍♂️ start predict speaker from: {start_time} to: {end_time}")

        try:
            audio_data, sample_rate = sf.read(segment_wav)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
        except Exception as e:
            print(f"Error loading audio file: {e}")

        predicted_speaker, confidence_scores = authenticator.authenticate_qcnn(
            audio_data=audio_data,
            sample_rate=sample_rate,
            model_dir="mfcc_qcnn_hmm_models"
        )

        segment_wav.seek(0)

        segment_file = FileStorage(
            stream=segment_wav,
            filename=f"segment_{start_time}_{end_time}.wav",
            content_type="audio/wav"
        )

        segment_path = Utils.store_WAV(segment_file)

        try:
            transcription = transcriber(segment_path)['text']

            results.append({
                "start_time": start_time,
                "end_time": end_time,
                "speaker_data": predicted_speaker,
                "transcription": transcription
            })
        finally:
            if os.path.exists(segment_path):
                os.unlink(segment_path)

    dialogue_text = "\n".join(
        f'{entry["speaker_data"]}: {entry["transcription"]}' for entry in results
    )

    prompt = "Đây là đoạn hội thoại theo format người nói: nội dung. HÃY TÓM TẮT LẠI THEO TỪNG NGƯỜI NÓI. " + dialogue_text

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    return jsonify({
        "message": "Diarization completed successfully.",
        "results": response.output_text,
    }), 200


def export_results_to_json(results, json_file_path):
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Successfully exported results to {json_file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the JSON file: {e}")


@app.route("/train-all-hmm", methods=["POST"])
def train_all_hmm():
    Trainner.train_hmm_model_all()
    return jsonify({"message": "Completed"})


@app.route("/train", methods=["POST"])
def train():
    if AUDIO not in request.files or SCRIPT not in request.files:
        return jsonify({ERROR: NO_FILES_PROVIDED}), 400

    user_dir = os.path.join(Config.TRAIN_VOICE, "")
    os.makedirs(user_dir, exist_ok=True)

    audio_file = request.files.get(AUDIO)
    script_file = request.files.get(SCRIPT)
    print("store data....")
    audio_path, script_path = Utils.store_data(audio_file, script_file)

    annotations = Utils.load_annotations(script_path)
    new_speaker_data = FeatureExtractor.extract_append_features(audio_path, annotations)

    for speaker, new_data in new_speaker_data.items():
        all_data = new_data.copy()
        for subfolder in [f.path for f in os.scandir(Config.TRAIN_VOICE) if f.is_dir()]:
            annotation_file = os.path.join(subfolder, Config.SCRIPT_FILE)
            audio_file = os.path.join(subfolder, Config.AUDIO_FILE)

            if os.path.exists(annotation_file) and os.path.exists(audio_file):
                existing_annotations = Utils.load_annotations(annotation_file)
                existing_speaker_data = FeatureExtractor.extract_append_features(audio_file, existing_annotations)

                if speaker in existing_speaker_data:
                    all_data.extend(existing_speaker_data[speaker])

        print("✨ train speaker: \t", speaker)
        ModelManager.train_hmm_model(speaker, all_data)

    return jsonify({MESSAGE: COMPLETED}), 200


@app.route("/predict", methods=["POST"])
def predict():
    global start, end, best_speaker, best_score, mfcc
    if AUDIO not in request.files or SCRIPT not in request.files:
        return jsonify({ERROR: NO_FILES_PROVIDED}), 400

    audio_file = request.files.get(AUDIO)
    script_file = request.files.get(SCRIPT)
    audio_path, script_path = Utils.store_data(audio_file, script_file)

    segments = []
    with open(script_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                start_time = Utils.parse_time(parts[0])
                end_time = Utils.parse_time((parts[1]))
                segments.append((start_time, end_time))

    y, sr = librosa.load(audio_path, sr=None)

    predictions = []

    for start, end in segments:
        segment = y[int(start * sr): int(end * sr)]
        mfcc = FeatureExtractor.extract_mfcc_from_segment(segment, sr)
        best_speaker = None
        best_score = float("-inf")

        for model_file in os.listdir(Config.MODELS_DIR):
            speaker = model_file.replace(MODEL_EXTENSION, "")
            with open(os.path.join(Config.MODELS_DIR, model_file), "rb") as f:
                model = pickle.load(f)

            score = model.score(mfcc)
            if score > best_score:
                best_score = score
                best_speaker = speaker

        predictions.append(f"{Utils.format_time(start)} {Utils.format_time(end)} {best_speaker}")

    with open(script_path, "w") as f:
        f.write("\n".join(predictions))

    return jsonify({MESSAGE: COMPLETED, OUTPUT: predictions}), 200


def merge_same_speaker_segments(results):
    if not results:
        return []

    merged_results = []
    current_segment = {
        "start_time": results[0]["start_time"],
        "end_time": results[0]["end_time"],
        "speaker_data": results[0]["speaker_data"],
        "transcription": results[0]["transcription"]
    }

    for i in range(1, len(results)):
        segment = results[i]
        if segment["speaker_data"] == current_segment["speaker_data"]:
            # Gộp đoạn lại
            current_segment["end_time"] = segment["end_time"]
            current_segment["transcription"] += " " + segment["transcription"]
        else:
            # Đẩy đoạn cũ vào danh sách kết quả
            merged_results.append(current_segment)
            # Bắt đầu đoạn mới
            current_segment = {
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "speaker_data": segment["speaker_data"],
                "transcription": segment["transcription"]
            }

    # Thêm đoạn cuối cùng
    merged_results.append(current_segment)
    return merged_results


@app.route("/predict_QCNN", methods=["POST"])
def predict_QCNN():
    import soundfile as sf
    import numpy as np

    audio_file = "./20_percent_test/test_segment_1.wav"

    try:
        audio_data, sample_rate = sf.read(audio_file)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        print(f"Error loading audio file: {e}")

    speaker, confidence = VoiceAuthenticator.authenticate_qcnn(audio_data, sample_rate)

    print(f"Predicted speaker: {speaker}")
    print("Confidence scores:")
    for s, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s}: {score:.4f}")


@app.route('/train-qcnn-hmm', methods=['POST'])
def train_model_qcnn_hmm():
    return jsonify({MESSAGE: COMPLETED}), 200


if __name__ == '__main__':
    app.run(debug=False)

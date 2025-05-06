import os
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


@app.route("/train-all-hmm", methods=["POST"])
def train_all_hmm():
    Trainner.train_hmm_model_all()
    return jsonify({"message": "Completed"})

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

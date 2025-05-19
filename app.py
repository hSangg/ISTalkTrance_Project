import io
import json
import os
import tempfile
import wave
from datetime import timedelta, datetime
from io import BytesIO

import librosa
import numpy as np
import pyannote.audio
import soundfile as sf
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from openai import OpenAI
from transformers import (pipeline)
from werkzeug.datastructures import FileStorage

from modules.MFCC_QCNN_HMM import SpeakerIdentification
from modules.authenticate import VoiceAuthenticator
from modules.batch_trainer import BatchTrainer
from modules.config import Config
from modules.trainner import Trainner
from modules.utils import Utils

load_dotenv()

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

mongo_url = os.getenv("MONGO_URI")
app.config["MONGO_URI"] = mongo_url
mongo = PyMongo(app)

hf_token_full_access = os.getenv("HF_TOKEN_FULL_ACCESS")
openai_token = os.getenv("OPENAI_API_KEY")
identification_threshold = os.getenv("IDENTIFICATION_THRESHOLD", 0.8)

authenticator = VoiceAuthenticator()
batch_trainer = BatchTrainer()
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-medium", token=hf_token_full_access)
pipeline = pyannote.audio.Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token_full_access)

class User:
    def __init__(self, email, name, picture, google_id):
        self.email = email
        self.name = name
        self.picture = picture
        self.google_id = google_id

    def to_dict(self):
        return {
            "email": self.email,
            "name": self.name,
            "picture": self.picture,
            "googleId": self.google_id
        }

class Room:
    def __init__(self, room_name, room_sid, summarization):
        self.room_name = room_name
        self.timestamp = datetime.utcnow()
        self.summarization = summarization
        self.room_sid = room_sid
        self.users = []

    def add_user(self, user):
        if isinstance(user, User):
            self.users.append(user)

    def to_dict(self):
        return {
            "roomName": self.room_name,
            "timestamp": self.timestamp.isoformat(),
            "summarization": self.summarization,
            "roomSid": self.room_sid,
            "users": [user.to_dict() for user in self.users]
        }


@app.route("/summarization", methods=["POST"])
def summarization():
    print("start summarization......")


    if AUDIO not in request.files:
        return jsonify({ERROR: NO_FILES_PROVIDED}), 400

    audio_file = request.files[AUDIO]

    audio_bytes = audio_file.read()

    audio_io = BytesIO(audio_bytes)

    with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
        wav_file.getframerate()
        wav_file.getnchannels()
        wav_file.getsampwidth()

    full_audio_io = BytesIO(audio_bytes)
    full_audio, sr = librosa.load(full_audio_io, sr=None, mono=True)

    diarization = pipeline(audio_io)

    results = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = str(timedelta(seconds=int(turn.start)))
        end_time = str(timedelta(seconds=int(turn.end)))

        if start_time == end_time:
            continue

        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)

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

        model_list_json = request.form.get("modelList")

        try:
            model_list = json.loads(model_list_json)
        except (json.JSONDecodeError, TypeError):
            model_list = None


        predicted_speaker, confidence_scores = authenticator.authenticate_qcnn(
            audio_data=audio_data,
            sample_rate=sample_rate,
            model_dir="mfcc_qcnn_hmm_models",
            model_list=model_list
        )

        segment_wav.seek(0)

        segment_file = FileStorage(
            stream=segment_wav,
            filename=f"segment_{start_time}_{end_time}.wav",
            content_type="audio/wav"
        )

        segment_path = Utils.store_WAV(segment_file)

        transcription = transcriber(segment_path)['text']

        results.append({
            "start_time": start_time,
            "end_time": end_time,
            "speaker_data": predicted_speaker,
            "transcription": transcription
        })

    dialogue_text = "\n".join(
        f'{entry["speaker_data"]} từ: {entry["start_time"]} đến: {entry["end_time"]} nói: {entry["transcription"]}' for entry in results
    )

    prompt = ("Đây là đoạn hội thoại theo định dạng theo chuẩn {người nói} từ {thời gian bắt đầu} đến: {thời gian kết thúc} nói: nội dung. HÃY TÓM TẮT LẠI THEO TỪNG NGƯỜI NÓI VÀ THỜi GIAN,"
              "LƯU Ý TRONG PHẦN NỘI DUNG CÓ THỂ SAI CHÍNH TẢ: ") + dialogue_text

    client = OpenAI(api_key=openai_token)

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    room_sid = request.form.get('roomSid')
    room_name = request.form.get('roomName')
    composition_sid = request.form.get('compositionSid')

    result = mongo.db.rooms.update_one(
        {"roomSid": room_sid},
        {"$set": {"summarization": response.output_text, "timestamp": datetime.utcnow()}}
    )

    if result.matched_count == 0:
        new_room = Room(room_name, room_sid, response.output_text)
        mongo.db.rooms.insert_one(new_room.to_dict())

    return jsonify({
        "message": "Summarization completed successfully.",
        "speech": results,
        "prompt": prompt,
        "summarization": response.output_text,
    }), 200


@app.route('/rooms', methods=['GET'])
def get_all_rooms():
    rooms = list(mongo.db.rooms.find({}).sort("timestamp", -1))
    return jsonify(rooms), 200

@app.route("/train-all-hmm", methods=["POST"])
def train_all_hmm():
    Trainner.train_hmm_model_all()
    return jsonify({"message": "Completed"})

@app.route("/predict_QCNN", methods=["POST"])
def predict_QCNN():
    if 'audio_file' not in request.files:
        return jsonify({"error": "Missing 'audio_file' in request"}), 400

    file = request.files['audio_file']

    try:
        # Đọc dữ liệu âm thanh từ file upload (dạng byte stream)
        audio_data, sample_rate = sf.read(io.BytesIO(file.read()))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # convert to mono nếu stereo
    except Exception as e:
        return jsonify({"error": f"Error reading audio file: {str(e)}"}), 500

    try:
        # Gọi hàm nhận diện giọng nói
        speaker, confidence = VoiceAuthenticator.authenticate_qcnn(audio_data, sample_rate)

        return jsonify({
            "predicted_speaker": speaker,
            "confidence": {s: float(score) for s, score in confidence.items()}
        })
    except Exception as e:
        return jsonify({"error": f"Model prediction error: {str(e)}"}), 500
    
MESSAGE = "message"
COMPLETED = "Training completed successfully"
ERROR = "Error during training"
MISSING_FILES = "Missing required files: script.txt and raw.wav"

@app.route('/train-qcnn-hmm', methods=['POST'])
def train_model_qcnn_hmm():
    try:
        if 'script' not in request.files or 'audio' not in request.files:
            return jsonify({MESSAGE: MISSING_FILES}), 400
        
        script_file = request.files['script']
        audio_file = request.files['audio']
        
        if script_file.filename == '' or audio_file.filename == '':
            return jsonify({MESSAGE: MISSING_FILES}), 400
        
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, 'script.txt')
            audio_path = os.path.join(temp_dir, 'raw.wav')
            
            script_file.save(script_path)
            audio_file.save(audio_path)
            
            import torch
            use_gpu = False
            
            si = SpeakerIdentification(use_gpu=use_gpu)
            
            speakers_data = si.parse_script(script_path)
            
            train_segments = {}
            for speaker, segments in speakers_data.items():
                train_segments[speaker] = [(audio_path, start, end) for start, end in segments]
            
            si.train(segments=train_segments)
            
            save_dir = si.save_dir
            
            stats = {
                "num_speakers": len(train_segments),
                "models_saved_at": save_dir,
                "gpu_used": use_gpu
            }
            
            return jsonify({
                MESSAGE: COMPLETED,
                "stats": stats
            }), 200
    
    except Exception as e:
        import traceback
        print(f"Error in train_model_qcnn_hmm: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            MESSAGE: ERROR,
            "error_details": str(e)
        }), 500

MODELS_DIR = "mfcc_qcnn_hmm_models"
@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        if not os.path.exists(MODELS_DIR):
            return jsonify({
                "status": "error",
                "message": f"Directory '{MODELS_DIR}' not found"
            }), 404

        models = [file for file in os.listdir(MODELS_DIR)
                  if os.path.isfile(os.path.join(MODELS_DIR, file)) and file.endswith('.pkl')]

        return jsonify({
            "status": "success",
            "count": len(models),
            "models": models
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=8080, debug=False)

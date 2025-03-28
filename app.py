from flask import Flask, request, jsonify, Blueprint
from modules.authenticate import VoiceAuthenticator
from modules.config import Config
from modules.batch_trainer import BatchTrainer
from werkzeug.utils import secure_filename
import os
from typing import Tuple, Dict, Any
import librosa
import numpy as np
import io

from modules.feature_extractor import FeatureExtractor
from modules.train import EnhancedBatchTrainer
from modules.utils import Utils
from modules.config import Config

app = Flask(__name__)

Config.setup()
from modules.train import EnhancedBatchTrainer

authenticator = VoiceAuthenticator()
batch_trainer = BatchTrainer()

@app.route('/train', methods=['POST'])
def train():
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        audio_file = request.files.get('audio_file')
        script_file = request.files.get('script_file')
        if not audio_file or not script_file:
            return jsonify({"error": "Both audio_file and script_file are required"}), 400

        user_dir = os.path.join("train_voice", user_id)
        os.makedirs(user_dir, exist_ok=True)

        audio_path = os.path.join(user_dir, "raw.wav")
        script_path = os.path.join(user_dir, "script.txt")
        audio_file.save(audio_path)
        script_file.save(script_path)

        trainer = EnhancedBatchTrainer(train_dir="train_voice")
        result = trainer.train_user_model(user_dir)

        if not result["success"]:
            os.remove(audio_path)
            os.remove(script_path)
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'wav', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/train/status', methods=['GET'])
def training_status():
    """Get the status of trained models"""
    try:
        models = authenticator.list_models()
        train_dir = 'train_data'
        
        if os.path.exists(train_dir):
            available_users = [d for d in os.listdir(train_dir) 
                             if os.path.isdir(os.path.join(train_dir, d))]
            
            return jsonify({
                "success": True,
                "trained_models": models["models"],
                "available_users": available_users,
                "trained_count": models["count"],
                "available_count": len(available_users),
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Training directory not found",
            }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500

@app.route('/train/batch', methods=['POST'])
def train_batch():
    """Train all models from the training data directory"""
    try:
        train_dir = request.json.get('train_dir', 'train_data') if request.is_json else 'train_data'
        
        result = batch_trainer.train_all(train_dir)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500

import time

@app.route('/authenticate', methods=['POST'])
def authenticate():
    start_time = time.time()

    if not request.files:
        return jsonify({"error": "No files provided"}), 400

    try:
        file_key, file = next(iter(request.files.items()))

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        audio_bytes = file.read()
        result = authenticator.authenticate(audio_bytes)

        elapsed_time = time.time() - start_time 

        response = {
            "elapsed_time": elapsed_time,
            **result
        }

        if result["success"]:
            return jsonify(response), 200
        else:
            return jsonify(response), 400

    except StopIteration:
        return jsonify({"error": "No files in the request"}), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/models', methods=['GET'])
def list_models():
    try:
        result = authenticator.list_models()
        return jsonify(result), 200 if result["success"] else 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    

def load_annotations(annotation_path):
    annotations = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_time = Utils.parse_time(parts[0])
                end_time = Utils.parse_time(parts[1])
                speaker = parts[2]
                annotations.append((start_time, end_time, speaker))
    return annotations

def process_folder(folder_path):
    speaker_data = {}
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for subfolder in subfolders:
        annotation_file = os.path.join(subfolder, "script.txt")
        audio_file = os.path.join(subfolder, "raw.wav")
        
        if not os.path.exists(annotation_file) or not os.path.exists(audio_file):
            continue
        
        annotations = load_annotations(annotation_file)
        folder_speaker_data = FeatureExtractor.extract_features(audio_file, annotations)
        
        for speaker, data in folder_speaker_data.items():
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].extend(data)
    
    for speaker, data in speaker_data.items():
        EnhancedBatchTrainer.train_hmm_model(speaker, data)

@app.route("/train", methods=["POST"])
def train_models():
    process_folder(Config.DATASET_PATH)
    return jsonify({"message": "Training completed"})


if __name__ == '__main__':
    app.run(debug=False)

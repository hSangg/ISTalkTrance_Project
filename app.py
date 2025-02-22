from flask import Flask, request, jsonify, Blueprint
from modules.authenticate import VoiceAuthenticator
from modules.config import Config
from modules.batch_trainer import BatchTrainer
from werkzeug.utils import secure_filename
from modules.qcnn_trainer import QCNNTrainer
from modules.qcnn_tester import QCNNTester
import os
from typing import Tuple, Dict, Any
import librosa
import numpy as np
import io

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

@app.route('/train/qcnn', methods=['POST'])
def train_qcnn():
    try:
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
            
        if 'audio_file' not in request.files or 'script_file' not in request.files:
            return jsonify({'error': 'Both audio and script files are required'}), 400
            
        audio_file = request.files['audio_file']
        script_file = request.files['script_file']
        
        if not allowed_file(audio_file.filename) or not allowed_file(script_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        upload_dir = os.path.join("uploads", secure_filename(user_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        audio_path = os.path.join(upload_dir, secure_filename(audio_file.filename))
        script_path = os.path.join(upload_dir, secure_filename(script_file.filename))
        
        audio_file.save(audio_path)
        script_file.save(script_path)
        
        trainer = QCNNTrainer()
        result = trainer.train_model(user_id, audio_path, script_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'wav', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/test/qcnn', methods=['POST'])
def test_qcnn():
    user_id = request.form['user_id']
    audio_file = request.files['test_audio']
    test_audio_path = os.path.join("uploads", "test_audio.wav")
    audio_file.save(test_audio_path)
    
    tester = QCNNTester(user_id)
    label = tester.predict_speaker(test_audio_path)
    
    return jsonify({"predicted_label": label})

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

@app.route('/authenticate', methods=['POST'])
def authenticate():
    if not request.files:
        return jsonify({"error": "No files provided"}), 400

    try:
        file_key, file = next(iter(request.files.items()))
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        audio_bytes = file.read()
        result = authenticator.authenticate(audio_bytes)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

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
    
if __name__ == '__main__':
    app.run(debug=False)

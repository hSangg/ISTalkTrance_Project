from flask import Flask, request, jsonify
from modules.authenticate import VoiceAuthenticator
# from modules.diarization import AutoSpectralDiarizer
from modules.config import Config
from modules.batch_trainer import BatchTrainer
import os
import numpy as np

import librosa
import numpy as np
import io

app = Flask(__name__)

Config.setup()
from train import EnhancedBatchTrainer

authenticator = VoiceAuthenticator()
batch_trainer = BatchTrainer()
# diarizer = AutoSpectralDiarizer()

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
    
@app.route('/diarization', methods=['POST'])
def diarization():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Create a temporary file to save the uploaded audio
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        OUTPUT_DIR = "separated_speakers"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        diarizer = AutoSpectralDiarizer()

        # Process the audio file
        speaker_audio, speaker_stats = diarizer.process_audio(temp_path, OUTPUT_DIR)

        # Clean up temporary file
        os.remove(temp_path)

        # Convert stats to serializable format
        speaker_stats_serializable = {
            speaker_id: {
                key: float(value) if isinstance(value, np.float32) else value
                for key, value in stats.items()
            }
            for speaker_id, stats in speaker_stats.items()
        }

        return jsonify({
            "success": True, 
            "stats": speaker_stats_serializable,
            "output_files": [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
        }), 200
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
    app.run(debug=True)
    
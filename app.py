import os

from flask import Flask, request, jsonify

from modules.authenticate import VoiceAuthenticator
from modules.batch_trainer import BatchTrainer
from modules.config import Config
from modules.feature_extractor import FeatureExtractor
from modules.utils import Utils

app = Flask(__name__)

Config.setup()
from modules.train_method import EnhancedBatchTrainer

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
        _, file = next(iter(request.files.items()))

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

@app.route("/train-all", methods=["POST"])
def train_all():
    speaker_data = {}
    subfolders = [f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()]

    for subfolder in subfolders:
        print("✨ start train for sub-folder \t", subfolder)
        annotation_file = os.path.join(subfolder, "script.txt")
        audio_file = os.path.join(subfolder, "raw.wav")

        if not os.path.exists(annotation_file) or not os.path.exists(audio_file):
            continue

        annotations = Utils.load_annotations(annotation_file)
        folder_speaker_data = FeatureExtractor.extract_features(audio_file, annotations)

        for speaker, data in folder_speaker_data.items():
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].extend(data)

    for speaker, data in speaker_data.items():
        print("✨ start train for \t", speaker)
        EnhancedBatchTrainer.train_hmm_model(speaker, data)
    return jsonify({"message": "Training completed"})

if __name__ == '__main__':
    app.run(debug=False)
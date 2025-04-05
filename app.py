import os
import pickle
import time

import librosa
from flask import Flask, request, jsonify

from modules.authenticate import VoiceAuthenticator
from modules.batch_trainer import BatchTrainer
from modules.config import Config
from modules.feature_extractor import FeatureExtractor
from modules.model_manager import ModelManager
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

Config.setup()
from modules.trainner import Trainner

authenticator = VoiceAuthenticator()
batch_trainer = BatchTrainer()

@app.route('/deprecated/train', methods=['POST'])
def deprecated_train():
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

        audio_path = os.path.join(user_dir, "raw.WAV")
        script_path = os.path.join(user_dir, "script.txt")
        audio_file.save(audio_path)
        script_file.save(script_path)

        trainer = Trainner(train_dir="train_voice")
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

@app.route('/authenticate', methods=['POST'])
def authenticate():
    start_time = time.time()

    if not request.files:
        return jsonify({"error": NO_FILES_PROVIDED}), 400

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

@app.route("/evaluation", methods=["POST"])
def evaluation():
    test_root = Config.TEST_VOICE
    model_dir = Config.MODELS_DIR

    for folder in os.listdir(test_root):
        folder_path = os.path.join(test_root, folder)
        if not os.path.isdir(folder_path):
            continue

        audio_path = next(
            (os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".wav")), None)
        script_path = next(
            (os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".txt")), None)

        if not audio_path or not script_path:
            print(f"❌ Thiếu file .wav hoặc .txt trong: {folder_path}")
            continue

        print(f"🔍 Đang test: {folder_path}")

        segments = []
        with open(script_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    start_time = Utils.parse_time(parts[0])
                    end_time = Utils.parse_time(parts[1])
                    segments.append((start_time, end_time))

        y, sr = librosa.load(audio_path, sr=None)

        predictions = []
        for start, end in segments:
            segment = y[int(start * sr): int(end * sr)]
            mfcc = FeatureExtractor.extract_mfcc_from_segment(segment, sr)
            if mfcc is None:
                predictions.append(f"{Utils.format_time(start)} {Utils.format_time(end)} Unknown")
                continue

            best_speaker = None
            best_score = float("-inf")

            for model_file in os.listdir(model_dir):
                if not model_file.endswith(".pkl"):
                    continue
                speaker = model_file.replace(".pkl", "")
                model_path = os.path.join(model_dir, model_file)

                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                try:
                    score = model.score(mfcc)
                    if score > best_score:
                        best_score = score
                        best_speaker = speaker
                except Exception as e:
                    print(f"⚠️ Lỗi khi predict với model {speaker}: {e}")

            predictions.append(f"{Utils.format_time(start)} {Utils.format_time(end)} {best_speaker or 'Unknown'}")

        result_path = os.path.join(folder_path, "script_predicted.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("\n".join(predictions))

        print(f"✅ Đã ghi kết quả vào {result_path}")

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
        segment = y[int (start * sr) : int(end * sr)]
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

if __name__ == '__main__':
    app.run(debug=False)
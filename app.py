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
hf_token = os.getenv("HF_TOKEN")
hf_token_full_access = os.getenv("HF_TOKEN_FULL_ACCESS")

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSeq2SeqLM
processor = Wav2Vec2Processor.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-vietnamese", token=hf_token_full_access)
model = Wav2Vec2ForCTC.from_pretrained("anuragshas/wav2vec2-large-xlsr-53-vietnamese", token=hf_token_full_access)

summary_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", token=hf_token_full_access)
summary_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base", token=hf_token_full_access)

from transformers import pipeline
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", token=hf_token_full_access)

# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# model_path = "vinai/PhoGPT-4B-Chat"
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, token="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")
# phoModel = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, token="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")
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token="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")
# PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"

#
# @app.route("/diarization", methods=["POST"])
# def diarization():
#     if AUDIO not in request.files:
#         return jsonify({ERROR: NO_FILES_PROVIDED}), 400
#
#     audio_file = request.files[AUDIO]
#     audio_bytes = audio_file.read()
#
#     audio_io = BytesIO(audio_bytes)
#
#     with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
#         framerate = wav_file.getframerate()
#         channels = wav_file.getnchannels()
#         sampwidth = wav_file.getsampwidth()
#
#     full_audio_io = BytesIO(audio_bytes)
#     full_audio, sr = librosa.load(full_audio_io, sr=None, mono=True)
#
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token)
#
#     print("prepare pipeline...")
#     diarization = pipeline(audio_io)
#
#     results = []
#
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         start_time = str(timedelta(seconds=int(turn.start)))
#         end_time = str(timedelta(seconds=int(turn.end)))
#
#         if start_time == end_time:
#             continue
#
#         start_sample = int(turn.start * sr)
#         end_sample = int(turn.end * sr)
#
#         audio_segment = full_audio[start_sample:end_sample]
#
#         segment_wav = BytesIO()
#         sf.write(segment_wav, audio_segment, sr, format='WAV')
#         segment_wav.seek(0)
#
#         print(f"🏃‍♂️ start predict speaker from: {start_time} to: {end_time}")
#         predict_speaker = authenticator.authenticate(segment_wav.read())
#         segment_wav.seek(0)
#         segment_file = FileStorage(
#             stream=segment_wav,
#             filename=f"segment_{start_time}_{end_time}.wav",
#             content_type="audio/wav"
#         )
#
#         segment_path = Utils.store_WAV(segment_file)
#
#         try:
#             transcription = transcriber(segment_path)['text']
#
#             result_line = f"{start_time} {end_time} {predict_speaker} {transcription}"
#             print(result_line.strip())
#
#             results.append({
#                 "start_time": start_time,
#                 "end_time": end_time,
#                 "speaker_data": predict_speaker["best_user"],
#                 "transcription": transcription
#             })
#         finally:
#             if os.path.exists(segment_path):
#                 os.unlink(segment_path)
#
#         dialogue_text = "\n".join(
#             f'{entry["speaker_data"]}: {entry["transcription"]}' for entry in results
#         )
#
#         instruction = "Đây là người nói và nội dung chưa đúng chính tả, hãy tổng hợp lại và tóm tắt cuộc họp " + dialogue_text
#         input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
#
#         input_ids = tokenizer(input_prompt, return_tensors="pt")
#
#         outputs = model.generate(
#             inputs=input_ids["input_ids"].to("cuda"),
#             attention_mask=input_ids["attention_mask"].to("cuda"),
#             do_sample=True,
#             temperature=1.0,
#             top_k=50,
#             top_p=0.9,
#             max_new_tokens=1024,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
#
#         response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         response = response.split("### Trả lời:")[1]
#
#         # input_text = f"tóm tắt: {dialogue_text}"
#         #
#         # input_ids = summary_tokenizer(
#         #     input_text,
#         #     return_tensors="pt",
#         #     max_length=512,
#         #     truncation=True
#         # ).input_ids
#         #
#         # output_ids = summary_model.generate(
#         #     input_ids,
#         #     max_length=150,
#         #     num_beams=4,
#         #     early_stopping=True
#         # )
#         # summary = summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return jsonify({
#         "message": "Diarization completed successfully.",
#         "results": response
#     }), 200


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

@app.route("/cross-validation", methods=["POST"])
def cross_validation():
    Trainner.cross_validate_all_speakers()
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
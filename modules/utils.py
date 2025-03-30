import io
import os
import pickle
import random
import string
from typing import List, Dict, Union

import numpy as np
import soundfile as sf
from hmmlearn import hmm

from modules.config import Config

LENGTH = 10

class Utils:
    @staticmethod
    def store_data(audio_file, script_file):
        dataset_path = os.path.join(Config.TRAIN_VOICE, Utils.generate_random_string(LENGTH))
        os.makedirs(dataset_path, exist_ok=True)

        audio_path = os.path.join(dataset_path, Config.AUDIO_FILE)
        script_path = os.path.join(dataset_path, Config.SCRIPT_FILE)
        audio_file.save(audio_path)
        script_file.save(script_path)

        return audio_path, script_path

    @staticmethod
    def train_hmm_model(speaker, data):
        model_path = f"{Config.MODELS_DIR}/{speaker}.pkl"

        if os.path.exists(model_path):
            os.remove(model_path)

        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=Config.HMM_ITERATIONS)
        X = np.vstack(data)
        lengths = [len(x) for x in data]
        model.fit(X, lengths)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def parse_time(time_str) -> float:
        h, m, s = map(float, time_str.split(":"))
        return h * 3600 + m * 60 + s

    @staticmethod
    def parse_timestamp_script(self, script_path: str) -> List[Dict[str, Union[float, str]]]:
        segments = []
        try:
            with open(script_path, 'r') as f:
                script_text = f.read().strip()
                time_labels = script_text.split()

            for i in range(0, len(time_labels), 3):
                start_time = Utils.parse_time(time_labels[i])
                end_time = Utils.parse_time(time_labels[i + 1])
                label = time_labels[i + 2]

                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': label
                })
            audio_path = os.path.join('train_voice', 'user123', 'raw.wav')
            print(f"Type of audio_path: {type(audio_path)}")

            output_dir = '20_percent_test'
            if segments:
                self.extract_and_export_20_percent(audio_path, segments, output_dir)
            else:
                print("No segments to extract.")
            return segments
        except Exception as e:
            print(f"Error parsing timestamp script: {e}")
            return []

    @staticmethod
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

    @staticmethod
    def generate_random_string(length=10):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))

        return random_string

    @staticmethod
    def validate_audio(audio_bytes):
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                return f.samplerate, f.channels
        except Exception as e:
            print(f"Invalid audio file: {e}")
            return None

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

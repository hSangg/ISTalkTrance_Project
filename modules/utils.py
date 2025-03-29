import os
from typing import List, Dict, Union

from modules.config import Config


class Utils:
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
    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

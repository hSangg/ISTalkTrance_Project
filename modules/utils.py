import os
import numpy as np
import librosa
import joblib
from typing import Dict, List

class Utils:
    @staticmethod
    def parse_time(time_str):
        h, m, s = map(float, time_str.split(":"))
        return h * 3600 + m * 60 + s

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
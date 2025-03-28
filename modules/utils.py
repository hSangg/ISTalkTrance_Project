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
import logging
import os


class Config:
    MODELS_DIR = './models'
    LOGGING_LEVEL = logging.INFO
    SAMPLE_RATE = 16000
    N_MFCC = 13
    HMM_COMPONENTS = 5
    HMM_ITERATIONS = 100
    TRAIN_VOICE = "./train_voice"
    ALLOWED_EXTENSIONS = {'wav', 'txt'}
    AUDIO_FILE = "raw.wav"
    SCRIPT_FILE = "script.txt"
    
    @staticmethod
    def setup():
        if not os.path.exists(Config.MODELS_DIR):
            os.makedirs(Config.MODELS_DIR)
        
        logging.basicConfig(
            level=Config.LOGGING_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

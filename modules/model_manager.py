import os
import joblib
from hmmlearn import hmm
import numpy as np
from .config import Config

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def save_model(self, user_id, model):
        """Save a model to disk"""
        try:
            model_path = os.path.join(Config.MODELS_DIR, f'{user_id}_model.pkl')
            joblib.dump(model, model_path)
            self.models[user_id] = model
            return True
        except Exception as e:
            return False

    def load_model(self, user_id):
        """Load a specific model from disk"""
        try:
            model_path = os.path.join(Config.MODELS_DIR, f'{user_id}_model.pkl')
            if os.path.exists(model_path):
                self.models[user_id] = joblib.load(model_path)
                return True
            return False
        except Exception as e:
            return False

    def load_all_models(self):
        """Load all models from disk"""
        self.models = {}
        if os.path.exists(Config.MODELS_DIR):
            for filename in os.listdir(Config.MODELS_DIR):
                if filename.endswith('_model.pkl'):
                    user_id = filename.replace('_model.pkl', '')
                    self.load_model(user_id)
    
    def train_model(self, user_id, features):
        """Train a new HMM model"""
        try:
            model = hmm.GaussianHMM(
                n_components=Config.HMM_COMPONENTS,
                covariance_type='diag',
                n_iter=Config.HMM_ITERATIONS
            )
            model.fit(features)
            self.save_model(user_id, model)
            return True
        except Exception as e:
            return False

    def get_model(self, user_id):
        """Get a model by user_id"""
        return self.models.get(user_id)

    def list_models(self):
        """List all available models"""
        return list(self.models.keys())
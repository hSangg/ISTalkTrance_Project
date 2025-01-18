import os
import joblib
import optuna
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
            print(f"Error saving model for user {user_id}: {e}")
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
            print(f"Error loading model for user {user_id}: {e}")
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
        """Train a new HMM model with parameter optimization using Optuna"""
        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, 10)
            covariance_type = trial.suggest_categorical("covariance_type", ["diag", "full", "tied", "spherical"])
            n_iter = trial.suggest_int("n_iter", 50, 200)

            model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
            try:
                model.fit(features)
                return model.score(features)
            except Exception:
                return float("-inf")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        best_params = study.best_params
        best_model = hmm.GaussianHMM(
            n_components=best_params["n_components"],
            covariance_type=best_params["covariance_type"],
            n_iter=best_params["n_iter"]
        )

        try:
            best_model.fit(features)
            self.save_model(user_id, best_model)
            return True
        except Exception as e:
            print(f"Error training model for user {user_id}: {e}")
            return False

    def get_model(self, user_id):
        """Get a model by user_id"""
        return self.models.get(user_id)

    def list_models(self):
        """List all available models"""
        return list(self.models.keys())
import os

import joblib
import numpy as np
import optuna
from hmmlearn import hmm
from sklearn.model_selection import KFold

from modules.config import Config


class ModelManager:
    def __init__(self, n_splits=5):
        self.models = {}
        self.load_all_models()
        self.n_splits = n_splits
        
    def save_model(self, user_id, model, cv_scores=None):
        try:
            model_data = {
                'model': model,
                'cv_scores': cv_scores
            }
            model_path = os.path.join(Config.MODELS_DIR, f'{user_id}_model.pkl')
            joblib.dump(model_data, model_path)
            self.models[user_id] = model
            return True
        except Exception as e:
            print(f"Error saving model for user {user_id}: {e}")
            return False

    def load_model(self, user_id):
        try:
            model_path = os.path.join(Config.MODELS_DIR, f'{user_id}_model.pkl')
            if os.path.exists(model_path):
                self.models[user_id] = joblib.load(model_path)
                return True
            return False
        except Exception as e:
            return False
        
    def load_all_models(self):
        self.models = {}
        if os.path.exists(Config.MODELS_DIR):
            for filename in os.listdir(Config.MODELS_DIR):
                if filename.endswith('_model.pkl'):
                    user_id = filename.replace('_model.pkl', '')
                    self.load_model(user_id)

    def cross_validate_model(self, model, features):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(features):
            train_features = features[train_idx]
            val_features = features[val_idx]
            
            try:
                fold_model = hmm.GaussianHMM(
                    n_components=model.n_components,
                    covariance_type=model.covariance_type,
                    n_iter=model.n_iter
                )
                
                fold_model.fit(train_features)
                
                score = fold_model.score(val_features)
                scores.append(score)
            except Exception:
                scores.append(float("-inf"))
                
        return np.mean(scores) if scores else float("-inf")

    def train_model(self, user_id, features):
        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, 10)
            covariance_type = trial.suggest_categorical("covariance_type", ["diag", "full", "tied", "spherical"])
            n_iter = trial.suggest_int("n_iter", 50, 200)

            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter
            )
            cv_score = self.cross_validate_model(model, features)
            return cv_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        best_params = study.best_params
        best_model = hmm.GaussianHMM(
            n_components=best_params["n_components"],
            covariance_type=best_params["covariance_type"],
            n_iter=best_params["n_iter"]
        )

        try:
            cv_scores = []
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(features):
                train_features = features[train_idx]
                val_features = features[val_idx]
                
                fold_model = hmm.GaussianHMM(
                    n_components=best_model.n_components,
                    covariance_type=best_model.covariance_type,
                    n_iter=best_model.n_iter
                )
                fold_model.fit(train_features)
                score = fold_model.score(val_features)
                cv_scores.append(score)
            
            best_model.fit(features)
            
            self.save_model(user_id, best_model, cv_scores)
            
            return {
                "success": True,
                "cv_scores": {
                    "mean": float(np.mean(cv_scores)),
                    "std": float(np.std(cv_scores)),
                    "scores": [float(score) for score in cv_scores]
                },
                "best_params": best_params
            }
        except Exception as e:
            print(f"Error training model for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_model(self, user_id):
        return self.models.get(user_id)

    def list_models(self):
        return list(self.models.keys())
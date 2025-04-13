import os

import joblib
import numpy as np
import optuna
from hmmlearn import hmm
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from modules.config import Config
from modules.utils import Utils


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
            model_path = os.path.join(Config.MODELS_DIR, f'{user_id}.pkl')
            if os.path.exists(model_path):
                self.models[user_id] = joblib.load(model_path)
                return True
            return False
        except Exception as e:
            return False
        
    def load_all_models(self):
        self.models = {}
        print("os.path.exists(Config.MODELS_DIR): ", os.path.exists(Config.MODELS_DIR))
        if os.path.exists(Config.MODELS_DIR):
            for filename in os.listdir(Config.MODELS_DIR):
                if filename.endswith('.pkl'):
                    user_id = filename.replace('.pkl', '')
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
    @staticmethod
    def predict_segment(segment, models):
        """
        Predict the speaker for a given feature segment using the trained HMM models.

        :param segment: The feature segment (e.g., MFCC) to be classified.
        :param models: A dictionary of trained HMM models for each speaker.

        :return: The predicted speaker label.
        """
        best_score = float("-inf")
        best_speaker = None

        for speaker, (model, _, _) in models.items():
            try:
                # Score the segment with the current model
                score = model.score(segment)
                if score > best_score:
                    best_score = score
                    best_speaker = speaker
            except Exception as e:
                print(f"❌ Error scoring segment for speaker {speaker}: {e}")

        return best_speaker

    def get_model(self, user_id):
        return self.models.get(user_id)

    def list_models(self):
        return list(self.models.keys())

    def cross_validate_hmm_model(speaker_data, n_splits=5):
        scores = {}

        for speaker, data in speaker_data.items():
            print(f"\n🔁 Cross-validating for speaker: {speaker}")
            folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            mfcc_data = data

            X_all = np.array(mfcc_data, dtype=object)
            fold_scores = []

            # Iterate over folds for cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X_all)):
                X_train = [X_all[i] for i in train_idx]
                X_test = [X_all[i] for i in test_idx]

                # Train HMM model
                model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=Config.HMM_ITERATIONS)
                train_concat = np.vstack(X_train)
                train_lengths = [len(x) for x in X_train]
                model.fit(train_concat, train_lengths)

                # Evaluate on test set
                test_concat = np.vstack(X_test)
                try:
                    y_true = [label for _, label in [X_all[i] for i in test_idx]]
                    y_pred = [ModelManager.predict_segment(feat, model) for feat, _ in [X_all[i] for i in test_idx]]

                    # Print Classification Report
                    print(f"📊 Fold {fold_idx + 1} Classification Report:")
                    print(classification_report(y_true, y_pred, zero_division=0))

                    fold_scores.append(classification_report(y_true, y_pred, output_dict=True))

                except Exception as e:
                    print(f"❌ Error scoring fold {fold_idx + 1}: {e}")

            # Store scores and print average score for each speaker
            scores[speaker] = fold_scores
            avg_score = np.mean([score['accuracy'] for score in fold_scores]) if fold_scores else float("-inf")
            print(f"\n📊 Average accuracy for {speaker}: {avg_score:.2f}")

        return scores


    @staticmethod
    def train_hmm_model(speaker, data):
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        model_path = f"{Config.MODELS_DIR}/{speaker}.pkl"

        if os.path.exists(model_path):
            os.remove(model_path)

        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=Config.HMM_ITERATIONS)
        X = np.vstack(data)
        lengths = [len(x) for x in data]
        model.fit(X, lengths)

        Utils.save_model(model, model_path)
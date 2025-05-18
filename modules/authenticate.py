import os
import pickle

import numpy as np

from modules.feature_extractor import FeatureExtractor
from modules.model_manager import ModelManager


class VoiceAuthenticator:

    def __init__(self):
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()

    def train(self, user_id, audio_files):
        try:
            all_features = []

            for audio_file in audio_files:
                audio_bytes = audio_file.read()
                features = self.feature_extractor.extract_mfcc_from_audio_bytes(audio_bytes)
                all_features.append(features)

            combined_features = np.concatenate(all_features)
            result = self.model_manager.train_model(user_id, combined_features)

            if result["success"]:
                return {
                    "success": True,
                    "message": f"Model trained successfully for user {user_id}",
                    "cv_scores": result["cv_scores"],
                    "best_params": result["best_params"]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error during training")
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def authenticate(self, audio_bytes):
        try:
            features = self.feature_extractor.extract_mfcc_from_audio_bytes(audio_bytes)
            scores = {}
            for user_id, model in self.model_manager.models.items():
                score = model.score(features)
                scores[user_id] = float(score)

            if not scores:
                return {
                    "success": False,
                    "error": "No models available for authentication",
                }

            best_user = max(scores, key=scores.get)

            return {
                "success": True,
                "best_user": best_user,
                "score": scores[best_user],
                "all_scores": scores,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @staticmethod
    def authenticate_qcnn(audio_data, sample_rate=16000, model_dir="mfcc_qcnn_hmm_models", model_list=None):
        try:

            with open(os.path.join(model_dir, "qcnn_weights.pkl"), "rb") as f:
                qcnn_weights = pickle.load(f)
        except FileNotFoundError:
            return "Error: Model files not found", {}

        hmm_models = {}
        for speaker in model_list:
            model_path = os.path.join(model_dir, f"{speaker}.pkl")
            try:
                with open(model_path, "rb") as f:
                    hmm_models[speaker] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Model for {speaker} not found")

        if not hmm_models:
            return "Error: No speaker models available", {}

        feature_extractor = FeatureExtractor(qcnn_weights)

        mfcc_features = feature_extractor.extract_mfcc(audio_data, sample_rate)

        if len(mfcc_features) == 0:
            return "Error: Failed to extract features", {}

        test_features = feature_extractor.process_qcnn(mfcc_features)

        scores = {}
        for speaker, model in hmm_models.items():
            try:
                score = model.score(test_features)
                scores[speaker] = score
            except Exception as e:
                print(f"Error scoring {speaker}: {e}")
                scores[speaker] = float('-inf')

        if not scores:
            return "unknown", {}

        predicted_speaker = max(scores.items(), key=lambda x: x[1])[0]

        max_score = max(scores.values())
        confidence_scores = {s: np.exp(score - max_score) for s, score in scores.items()}

        return predicted_speaker, confidence_scores

    def list_models(self):
        try:
            models = self.model_manager.list_models()
            return {
                "success": True,
                "models": models,
                "count": len(models),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

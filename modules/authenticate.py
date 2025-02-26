from .feature_extractor import FeatureExtractor
from .model_manager import ModelManager
from .config import Config
import numpy as np

class VoiceAuthenticator:
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.feature_extractor = FeatureExtractor()
        
    def train(self, user_id, audio_files):
        """Train a model for a user using multiple audio files"""
        try:
            all_features = []
            
            for audio_file in audio_files:
                audio_bytes = audio_file.read()
                features = self.feature_extractor.extract_mfcc_from_bytes(audio_bytes)
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
        """Authenticate a user based on voice sample"""
        try:
            features = self.feature_extractor.extract_mfcc_from_bytes(audio_bytes)
            
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

    def list_models(self):
        """List all available models"""
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
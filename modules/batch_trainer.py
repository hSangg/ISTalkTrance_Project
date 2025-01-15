import os
import numpy as np
from .feature_extractor import FeatureExtractor
from .model_manager import ModelManager

class BatchTrainer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_manager = ModelManager()
        
    def process_user_directory(self, user_id, user_dir):
        """Process all audio files for a single user"""
        try:
            all_features = []
            audio_files = [f for f in os.listdir(user_dir) if f.endswith(('.wav', '.WAV'))]
            
            if not audio_files:
                return {
                    "success": False,
                    "user_id": user_id,
                    "error": "No audio files found",
                    "processed_files": 0
                }
            
            for audio_file in audio_files:
                file_path = os.path.join(user_dir, audio_file)
                try:
                    with open(file_path, 'rb') as f:
                        audio_bytes = f.read()
                        features = self.feature_extractor.extract_mfcc_from_bytes(audio_bytes)
                        all_features.append(features)
                except Exception as e:
                    continue
            
            if all_features:
                combined_features = np.concatenate(all_features)
                success = self.model_manager.train_model(user_id, combined_features)
                
                return {
                    "success": success,
                    "user_id": user_id,
                    "processed_files": len(all_features),
                    "error": None if success else "Model training failed"
                }
            else:
                return {
                    "success": False,
                    "user_id": user_id,
                    "error": "No features could be extracted",
                    "processed_files": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "user_id": user_id,
                "error": str(e),
                "processed_files": 0
            }

    def train_all(self, train_data_dir='train_data'):
        """Train models for all users in the training directory"""
        if not os.path.exists(train_data_dir):
            return {
                "success": False,
                "error": f"Training directory {train_data_dir} does not exist",
            }

        results = []
        total_success = 0
        users_processed = 0

        for user_id in os.listdir(train_data_dir):
            user_dir = os.path.join(train_data_dir, user_id)
            if os.path.isdir(user_dir):
                users_processed += 1
                result = self.process_user_directory(user_id, user_dir)
                results.append(result)
                if result["success"]:
                    total_success += 1

        return {
            "success": total_success > 0,
            "total_users": users_processed,
            "successful_trainings": total_success,
            "failed_trainings": users_processed - total_success,
            "details": results,
        }

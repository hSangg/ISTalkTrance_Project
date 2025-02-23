import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from typing import Tuple, Dict, List, Any
from werkzeug.utils import secure_filename
import logging
from tensorflow.keras import layers, models
from utils.audio_processing import extract_mfcc_features
from sklearn.model_selection import StratifiedKFold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QCNNTrainer:
    def __init__(self):
        """Initialize QCNNTrainer with model directory and speaker mapping."""
        self.model_dir = "models_qcnn"
        self.speaker_map = {}  # Map speaker names to integer labels
        os.makedirs(self.model_dir, exist_ok=True)

    def build_qcnn_model(self, input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
        """Build and compile a QCNN model with 3 convolution layers."""
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def preprocess_data(self, mfcc_features: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess MFCC features and labels, returning padded/truncated features and numeric labels."""
        try:
            # Validate inputs
            if not mfcc_features or not labels:
                raise ValueError("Empty features or labels provided")
            if len(mfcc_features) != len(labels):
                raise ValueError(f"Mismatched features ({len(mfcc_features)}) and labels ({len(labels)})")
            
            # Create speaker mapping
            unique_speakers = sorted(set(labels))
            self.speaker_map = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
            
            # Convert labels to numeric
            numeric_labels = np.array([self.speaker_map[label] for label in labels])
            
            # Find dimensions of MFCC features and calculate target length (median length of sequences)
            n_mfcc = mfcc_features[0].shape[1]
            lengths = [mfcc.shape[0] for mfcc in mfcc_features]
            target_len = int(np.median(lengths))  # Use median length as target
            logger.info(f"Target sequence length: {target_len}")
            
            # Pad or truncate sequences to make them uniform
            processed_features = []
            for mfcc in mfcc_features:
                current_len = mfcc.shape[0]
                if current_len > target_len:
                    # Truncate: take the middle portion
                    start = (current_len - target_len) // 2
                    processed = mfcc[start:start + target_len, :]
                else:
                    # Pad: add zeros at both ends
                    pad_length = target_len - current_len
                    pad_left = pad_length // 2
                    pad_right = pad_length - pad_left
                    processed = np.pad(mfcc, ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                processed_features.append(processed)
            
            X = np.array(processed_features)
            logger.info(f"Final data shape: {X.shape}, Labels shape: {numeric_labels.shape}")
            return X, numeric_labels
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_model(self, user_id: str, audio_path: str, script_path: str) -> Dict[str, Any]:
        """Train QCNN model with cross-validation and save it along with preprocessing parameters."""
        try:
            # Feature extraction
            logger.info("Starting feature extraction...")
            mfcc_features, labels = extract_mfcc_features(audio_path, script_path)
            
            if not mfcc_features or not labels:
                raise ValueError("Feature extraction failed - no features or labels returned")
            
            # Data preprocessing
            logger.info("Preprocessing data...")
            X, y = self.preprocess_data(mfcc_features, labels)

            # Initialize cross-validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_results = []

            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                logger.info(f"Training fold {fold + 1}...")
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Build and train model
                model = self.build_qcnn_model(X_train.shape[1:], len(set(y)))
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                ]
                history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

                # Evaluate on validation data
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                fold_results.append((val_loss, val_acc))
                logger.info(f"Fold {fold + 1} - Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

            # Calculate average results
            avg_val_acc = np.mean([result[1] for result in fold_results])
            avg_val_loss = np.mean([result[0] for result in fold_results])
            logger.info(f"Average Validation Accuracy: {avg_val_acc:.4f}, Average Validation Loss: {avg_val_loss:.4f}")

            # Save the final model and parameters
            model_dir = os.path.join(self.model_dir, secure_filename(user_id))
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.h5")
            params_path = os.path.join(model_dir, "params.pkl")
            
            model.save(model_path)
            with open(params_path, 'wb') as f:
                pickle.dump({'speaker_map': self.speaker_map, 'input_shape': X_train.shape[1:]}, f)

            return {
                "success": True,
                "model_path": model_path,
                "validation_accuracy": float(avg_val_acc),
                "validation_loss": float(avg_val_loss),
                "num_classes": len(set(y)),
                "training_samples": len(X),
                "feature_shape": X_train.shape[1:]
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise Exception(f"Training failed: {str(e)}")

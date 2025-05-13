import warnings
import os
import pickle
import json
import torch
import numpy as np
import pennylane as qml
from hmmlearn import hmm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time
from datetime import datetime
import librosa
from modules.feature_extractor import FeatureExtractor

warnings.filterwarnings('ignore')

class SpeakerIdentification:
    def __init__(self, n_qubits=7, n_hmm_components=1, use_gpu=True):
        self.n_qubits = n_qubits
        self.n_hmm_components = n_hmm_components
        self.speakers = {}
        self.hmm_models = {}
        self.save_dir = "mfcc_qcnn_hmm_models"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            print(f"GPU is available. Using: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU instead.")
        
        self.n_mfcc = 20
        self.n_fft = 2048
        self.hop_length = 512
        
        dev = qml.device("default.qubit" if self.use_gpu else "default.qubit", wires=n_qubits)
        @qml.qnode(dev)
        def qcnn(inputs, weights, n_qubits=4, num_layers=2):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i], wires=i)
            
            for layer in range(num_layers):
                for i in range(n_qubits-1):
                    qml.CRZ(weights[layer * (n_qubits-1) + i], wires=[i, i+1])
                for i in range(n_qubits):
                    weight_idx = layer * n_qubits + i
                    if weight_idx < len(weights):
                        qml.RY(weights[weight_idx], wires=i)
            
            for i in range(0, n_qubits, 2):
                if i + 1 < n_qubits:
                    qml.CNOT(wires=[i, i+1])
                    qml.RY(weights[-1] * np.pi, wires=i+1)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qcnn = qcnn
        self.weights = np.full(n_qubits * 2 + 1, 0.5)

    def extract_mfcc(self, audio_path, start_time, end_time):
        """Extract MFCC features from audio segment"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            audio_segment = y[start_sample:end_sample]
            
            mfccs = librosa.feature.mfcc(
                y=audio_segment, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            mfcc_features = np.concatenate([mfcc_mean, mfcc_std])
            
            return mfcc_features
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return np.array([])


    def parse_script(self, script_path):
        speakers_local = {}
        
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_time = self.time_to_seconds(parts[0])
                end_time = self.time_to_seconds(parts[1])
                speaker = parts[2]
                
                if speaker not in speakers_local:
                    speakers_local[speaker] = []
                speakers_local[speaker].append((start_time, end_time))
        
        return speakers_local

    def time_to_seconds(self, time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s



    def train(self, segments=None):
        """
        Train speaker identification models with feature caching
        
        Parameters:
        -----------
        segments : dict, optional
            Dictionary mapping speaker names to lists of (audio_path, start_time, end_time) tuples
        """
        train_data = segments if segments is not None else self.speakers
        
        print(f"Starting training... {len(train_data)} speakers to process")
        os.makedirs(self.save_dir, exist_ok=True)
        
        features_dir = os.path.join(self.save_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        # Define a proper feature extractor using the QCNN weights
        feature_extractor = FeatureExtractor(self.weights, n_qubits=self.n_qubits)

        for speaker, segments in train_data.items():
            print(f"\n🔹 Speaker: {speaker}, Expected Segments: {len(segments)}")
            all_features = []
            
            feature_path = os.path.join(features_dir, f"{speaker}_features.pkl")
            if os.path.exists(feature_path):
                print(f"Loading existing features for {speaker} from {feature_path}")
                try:
                    with open(feature_path, "rb") as f:
                        existing_features = pickle.load(f)
                        all_features = existing_features
                    print(f"Loaded {len(all_features)} existing feature sets")
                except Exception as e:
                    print(f"Error loading existing features: {e}")
            
            start_time = time.time()
            new_features_count = 0
            
            for audio_path, start, end in segments:
                # Load audio segment
                y, sr = librosa.load(audio_path, sr=16000)
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                audio_segment = y[start_sample:end_sample]
                
                # Extract and process features with the same method that will be used during authentication
                mfcc_features = feature_extractor.extract_mfcc(audio_segment, sr)
                
                if len(mfcc_features) > 0:
                    q_features = feature_extractor.process_qcnn(mfcc_features)
                    if q_features.shape[0] > 0:
                        all_features.append(q_features)
                        new_features_count += 1
                
            end_time = time.time()
            print(f"Feature extraction and transformation took {end_time - start_time:.2f} seconds.")
            print(f"Added {new_features_count} new feature sets")
            
            if new_features_count > 0:
                with open(feature_path, "wb") as f:
                    pickle.dump(all_features, f)
                print(f"Saved combined features to {feature_path}")

            if not all_features:
                print(f"⚠️ No features extracted for {speaker}, skipping...")
                continue
                
            features = np.vstack(all_features)
            
            print(f"Training HMM for {speaker} with {features.shape} data points...")
            
            model = hmm.GaussianHMM(n_components=self.n_hmm_components, 
                                covariance_type="diag", 
                                n_iter=100, verbose=False)
            model.fit(features)
            self.hmm_models[speaker] = model
            
            model_path = os.path.join(self.save_dir, f"{speaker}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✅ Saved HMM model for {speaker} at {model_path}")

            weights_path = os.path.join(self.save_dir, "qcnn_weights.pkl")
            with open(weights_path, "wb") as f:
                pickle.dump(self.weights, f)

            speakers_list_path = os.path.join(self.save_dir, "speakers.json")
            if os.path.exists(speakers_list_path):
                with open(speakers_list_path, "r") as f:
                    speakers_list = json.load(f)
            else:
                speakers_list = []

            if speaker not in speakers_list:
                speakers_list.append(speaker)

            with open(speakers_list_path, "w") as f:
                json.dump(speakers_list, f)

    def predict(self, test_audio_path, start_time, end_time):
        """
        Predict speaker from audio segment
        
        Parameters:
        -----------
        test_audio_path : str
            Path to the audio file
        start_time : float
            Start time of the segment in seconds
        end_time : float
            End time of the segment in seconds
            
        Returns:
        --------
        str
            Predicted speaker
        dict
            Confidence scores for each speaker
        """
        # Load audio segment
        y, sr = librosa.load(test_audio_path, sr=16000)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = y[start_sample:end_sample]
        
        # Create feature extractor with the saved weights
        feature_extractor = FeatureExtractor(self.weights, n_qubits=self.n_qubits)
        
        # Extract features using the same process as during training
        mfcc_features = feature_extractor.extract_mfcc(audio_segment, sr)
        
        if len(mfcc_features) == 0:
            print(f"⚠️ Failed to extract MFCC features from {test_audio_path} at {start_time}-{end_time}")
            # Return default prediction if extraction fails
            return list(self.hmm_models.keys())[0] if self.hmm_models else "unknown", {}
        
        test_features = feature_extractor.process_qcnn(mfcc_features)
        
        scores = {}
        for speaker, model in self.hmm_models.items():
            try:
                score = model.score(test_features)
                scores[speaker] = score
            except Exception as e:
                print(f"Error scoring {speaker}: {e}")
                scores[speaker] = float('-inf')
        
        if not scores:
            return "unknown", {}
            
        predicted_speaker = max(scores.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence scores
        max_score = max(scores.values())
        confidence_scores = {s: np.exp(score - max_score) for s, score in scores.items()}
        
        return predicted_speaker, confidence_scores

def save_model(si, save_dir="mfcc_qcnn_hmm_models"):
    os.makedirs(save_dir, exist_ok=True)
    
    for speaker, model in si.hmm_models.items():
        model_path = os.path.join(save_dir, f"{speaker}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    weights_path = os.path.join(save_dir, "qcnn_weights.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump(si.weights, f)
    
    speakers_path = os.path.join(save_dir, "speakers.json")
    with open(speakers_path, "w") as f:
        json.dump(list(si.hmm_models.keys()), f)
    
    print(f"✅ Models saved in {save_dir}")

def cross_validate(base_folder, k=3, save_dir="crossval_models", log_file="crossval_log.txt", use_gpu=True):
    os.makedirs(save_dir, exist_ok=True)

    def append_log(lines):
            with open(log_file, "a", encoding="utf-8") as f:
                for line in lines:
                    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                    f.write(f"{timestamp} {line}\n")

    si_base = SpeakerIdentification(use_gpu=use_gpu)

    datasets = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    all_segments = {}

    for dataset in datasets:
        dataset_path = os.path.join(base_folder, dataset)

        audio_file = None
        for ext in ["raw.wav", "raw.WAV"]:
            audio_file_candidate = os.path.join(dataset_path, ext)
            if os.path.exists(audio_file_candidate):
                audio_file = audio_file_candidate
                break

        script_file = os.path.join(dataset_path, "script.txt")

        if not audio_file or not os.path.exists(script_file):
            print(f"Skipping {dataset}: Missing raw.wav/raw.WAV or script.txt")
            continue

        print(f"Loading dataset: {dataset} with audio file {audio_file}")

        speakers_data = si_base.parse_script(script_file)
        for speaker, segments in speakers_data.items():
            if speaker not in all_segments:
                all_segments[speaker] = []
            all_segments[speaker].extend([(audio_file, start, end) for start, end in segments])

    metrics = []

    for fold in range(k):
        fold_log = [f"\n📂 Fold {fold+1}/{k}"]
        train_segments = {}
        test_segments = {}

        for speaker, segments in all_segments.items():
            if len(segments) < k:
                fold_log.append(f"⚠️ Skipping speaker {speaker} with only {len(segments)} segments (need at least {k})")
                continue
                
            kf = KFold(n_splits=k, shuffle=True, random_state=fold)
            splits = list(kf.split(segments))

            train_idx, test_idx = splits[fold]
            train_segments[speaker] = [segments[i] for i in train_idx]
            test_segments[speaker] = [segments[i] for i in test_idx]

        if not train_segments:
            fold_log.append("⚠️ No speakers with enough data for this fold, skipping")
            append_log(fold_log)
            continue

        fold_dir = os.path.join(save_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        si = SpeakerIdentification(use_gpu=use_gpu)
        si.speakers = train_segments
        si.train()
        save_model(si, save_dir=fold_dir)

        total = 0
        correct = 0
        y_true = []
        y_pred = []

        for speaker, segments in test_segments.items():
            for audio_path, start, end in segments:
                predicted_speaker, _ = si.predict(audio_path, start, end)
                y_true.append(speaker)
                y_pred.append(predicted_speaker)
                if predicted_speaker == speaker:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        fold_log.append(f"Accuracy: {accuracy:.2%}")
        fold_log.append(f"Macro - Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1-score: {f1_macro:.2f}")
        fold_log.append(f"Weighted - Precision: {precision_weighted:.2f}, Recall: {recall_weighted:.2f}, F1-score: {f1_weighted:.2f}")

        append_log(fold_log)

        metrics.append((accuracy, precision_macro, recall_macro, f1_macro))

    if metrics:
        avg_metrics = np.mean(metrics, axis=0)
        summary_lines = [
            "\n📊 Cross-Validation Summary:",
            f"Average Accuracy: {avg_metrics[0]:.2%}",
            f"Average Precision: {avg_metrics[1]:.2f}",
            f"Average Recall: {avg_metrics[2]:.2f}",
            f"Average F1-score: {avg_metrics[3]:.2f}"
        ]
        append_log(summary_lines)
    
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected, falling back to CPU")
    
    train_voice_folder = "../reserve"
    cross_validate(train_voice_folder, k=3, use_gpu=use_gpu)
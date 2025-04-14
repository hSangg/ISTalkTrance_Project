import warnings
import os
import pickle
import json

import librosa
import numpy as np
import pennylane as qml
from hmmlearn import hmm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
warnings.filterwarnings('ignore')

class SpeakerIdentification:
    def __init__(self, n_mfcc=13, n_qubits=4, n_hmm_components=5, use_gpu=True):
        self.n_mfcc = n_mfcc
        self.n_qubits = n_qubits
        self.n_hmm_components = n_hmm_components
        self.speakers = {}
        self.hmm_models = {}
        self.save_dir = "mfcc_qcnn_hmm_single"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            print(f"GPU is available. Using: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU instead.")
        
        dev = qml.device("lightning.gpu", wires=n_qubits)
        @qml.qnode(dev)
        def qcnn(inputs, weights, n_qubits=2, num_layers=2):
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
        self.weights = np.random.randn(n_qubits)

    def extract_mfcc(self, audio_path, start_time, end_time):

        y, sr = librosa.load(audio_path, sr=None)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        audio_segment = y[start_sample:end_sample]
        
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=self.n_mfcc)
        
        return mfcc.T

    def process_qcnn(self, mfcc_features, step=5):
        quantum_features = []
        for i in range(0, len(mfcc_features), step):
            frame = mfcc_features[i]
            if len(frame) < self.n_qubits:
                frame = np.pad(frame, (0, self.n_qubits - len(frame)))
            else:
                frame = frame[:self.n_qubits]
            
            q_features = self.qcnn(frame, self.weights)
            quantum_features.append(q_features)
        
        return np.array(quantum_features)

    def parse_script(self, script_path):
        speakers_local = {}
        
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
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

    def train_test_split(self):
        train_data = {}
        test_data = {}
        
        for speaker, segments in self.speakers.items():
            if len(segments) > 1:
                train_segments, test_segments = train_test_split(segments, test_size=0.2)
                train_data[speaker] = train_segments
                test_data[speaker] = test_segments
        
        return train_data, test_data

    def train(self):
        train_data, _ = self.train_test_split()
        
        print(f"Starting training... {len(train_data)} speakers to process")
        os.makedirs(self.save_dir, exist_ok=True)
    
        for speaker, segments in train_data.items():
            print(f"\n🔹 Speaker: {speaker}, Expected Segments: {len(segments)}")
            all_features = []
            start_time = time.time()
            q_features = self.batch_process_features(segments)
            end_time = time.time()
            print(f"Feature extraction and transformation took {end_time - start_time:.2f} seconds.")

            if len(q_features) > 0:
                all_features.append(q_features)

            
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
        test_mfcc = self.extract_mfcc(test_audio_path, start_time, end_time)
        test_features = self.process_qcnn(test_mfcc)
        
        scores = {}
        for speaker, model in self.hmm_models.items():
            score = model.score(test_features)
            scores[speaker] = score
        
        predicted_speaker = max(scores.items(), key=lambda x: x[1])[0]
        return predicted_speaker, scores
    
    def evaluate(self, audio_path):
        total = 0
        correct = 0
        y_true = []
        y_pred = []
        
        _, test_data = self.train_test_split()
        
        print("\nTest Results:")
        print("Segment\tTrue Speaker\tPredicted Speaker")
        print("-" * 50)
        
        for speaker, segments in test_data.items():
            for start, end in segments:
                predicted_speaker, _ = self.predict(audio_path, start, end)
                print(f"{start}-{end}\t{speaker}\t{predicted_speaker}")
                
                y_true.append(speaker)
                y_pred.append(predicted_speaker)
                
                if predicted_speaker == speaker:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nEvaluation - Accuracy: {accuracy:.2%}")
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score (Macro-Averaged): {f1:.2f}")
        
        return precision, recall, f1

    def batch_process_features(self, segments_batch, batch_size=32):
        """Process features in batches to better utilize GPU memory"""
        all_features = []
        
        for audio_path, start, end in segments_batch:
            mfcc = self.extract_mfcc(audio_path, start, end)
            
            if self.use_gpu and len(mfcc) > 0:
                frames = []
                for i in range(0, len(mfcc), 3):
                    frame = mfcc[i]
                    if len(frame) < self.n_qubits:
                        frame = np.pad(frame, (0, self.n_qubits - len(frame)))
                    else:
                        frame = frame[:self.n_qubits]
                    frames.append(frame)
                
                if frames:
                    frames_tensor = torch.tensor(np.array(frames), device=self.device, dtype=torch.float32)
                    dataset = TensorDataset(frames_tensor)
                    dataloader = DataLoader(dataset, batch_size=batch_size)
                    
                    quantum_features = []
                    for batch in dataloader:
                        batch_frames = batch[0].cpu().numpy()
                        for frame in batch_frames:
                            q_features = self.qcnn(frame, self.weights)
                            quantum_features.append(q_features)
                    
                    all_features.append(np.array(quantum_features))
            else:
                q_features = self.process_qcnn(mfcc)
                if len(q_features) > 0:
                    all_features.append(q_features)
        
        if all_features:
            return np.vstack(all_features)
        return np.array([])

def train_all_datasets(base_folder, use_gpu=True):
    si = SpeakerIdentification(use_gpu=use_gpu)
    
    datasets = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    all_segments = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(base_folder, dataset)
        audio_file = os.path.join(dataset_path, "raw.wav")
        script_file = os.path.join(dataset_path, "script.txt")
        
        if not os.path.exists(audio_file) or not os.path.exists(script_file):
            print(f"Skipping {dataset}: Missing raw.wav or script.txt")
            continue
        
        print(f"Loading dataset: {dataset}")
        
        speakers_data = si.parse_script(script_file)
        for speaker, segments in speakers_data.items():
            if speaker not in all_segments:
                all_segments[speaker] = []
            all_segments[speaker].extend([(audio_file, start, end) for start, end in segments])


    print("\nTraining on all datasets combined...")

    si.speakers = all_segments
    si.train()

    return si

def evaluate_all_datasets(si, base_folder):
    print("\nEvaluating on all datasets...")
    
    datasets = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    total = 0
    correct = 0
    y_true = []
    y_pred = []
    
    for dataset in datasets:
        dataset_path = os.path.join(base_folder, dataset)
        audio_file = os.path.join(dataset_path, "raw.wav")
        script_file = os.path.join(dataset_path, "script.txt")
        
        if not os.path.exists(audio_file) or not os.path.exists(script_file):
            continue
        
        speakers_data = si.parse_script(script_file)
        
        for speaker, segments in speakers_data.items():
            for start, end in segments:
                predicted_speaker, _ = si.predict(audio_file, start, end)

                y_true.append(speaker)
                y_pred.append(predicted_speaker)

                if predicted_speaker == speaker:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Evaluation - Accuracy: {accuracy:.2%}")
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score (Macro-Averaged): {f1:.2f}")
    
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

def load_model(si, load_dir="mfcc_qcnn_hmm_models"):
    
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Directory {load_dir} does not exist.")
    
    speakers_path = os.path.join(load_dir, "speakers.json")
    
    if not os.path.exists(speakers_path):
        raise FileNotFoundError(f"File {speakers_path} not found.")
    
    with open(speakers_path, "r") as f:
        try:
            speakers = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {speakers_path}.")
    
    if not speakers:
        raise ValueError(f"No speakers found in {speakers_path}.")
    
    for speaker in speakers:
        model_path = os.path.join(load_dir, f"{speaker}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for speaker {speaker} not found: {model_path}")
        
        with open(model_path, "rb") as f:
            try:
                si.hmm_models[speaker] = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load HMM model for {speaker}: {e}")
    
    weights_path = os.path.join(load_dir, "qcnn_weights.pkl")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"QCNN weights file not found: {weights_path}")
    
    with open(weights_path, "rb") as f:
        try:
            si.weights = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load QCNN weights: {e}")
    
    print(f"✅ Models successfully loaded from {load_dir}")
    return si
def cross_validate(base_folder, k=3, save_dir="crossval_models", log_file="crossval_log.txt", use_gpu=True):
    os.makedirs(save_dir, exist_ok=True)

    def append_log(lines):
        with open(log_file, "a") as f:
            for line in lines:
                f.write(line + "\n")

    with open(log_file, "w") as f:
        f.write(f"🔁 Cross-Validation Start - {k} folds\n")

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
            if len(segments) < k + 1:
                continue
            kf = KFold(n_splits=k, shuffle=True, random_state=fold)
            splits = list(kf.split(segments))

            train_idx, test_idx = splits[fold]
            train_segments[speaker] = [segments[i] for i in train_idx]
            test_segments[speaker] = [segments[i] for i in test_idx]

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
    
    train_voice_folder = "../../train_voice"
    cross_validate(train_voice_folder, k=3, use_gpu=use_gpu)

import warnings

import numpy as np
import pennylane as qml
import torch
import torchaudio
from hmmlearn import hmm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from speechbrain.inference.classifiers import EncoderClassifier

warnings.filterwarnings('ignore')

class SpeakerIdentification:
    def __init__(self, n_qubits=4, n_hmm_components=2):
        self.n_qubits = n_qubits
        self.n_hmm_components = n_hmm_components
        self.speakers = {}
        self.hmm_models = {}
        
        # Initialize quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for i in range(n_qubits):
                qml.CRZ(weights[i], wires=[i, (i+1)%n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.qcnn = quantum_circuit
        self.weights = np.random.randn(n_qubits)
        
        # Load pretrained x-vector model
        self.embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/xvector"
        )

    def extract_xvector(self, audio_path, start_time, end_time):
        # Load audio segment
        y, sr = torchaudio.load(audio_path)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[:, start_sample:end_sample]
        
        # Extract x-vector embedding
        with torch.no_grad():
            embeddings = self.embedding_model.encode_batch(segment).squeeze(0)
        return embeddings.squeeze(0).numpy()

    def process_qcnn(self, xvector):
        # Process x-vector with quantum circuit
        if len(xvector) < self.n_qubits:
            xvector = np.pad(xvector, (0, self.n_qubits - len(xvector)))
        else:
            xvector = xvector[:self.n_qubits]
        
        q_features = self.qcnn(xvector, self.weights)
        return np.array(q_features)

    def parse_script(self, script_path):
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            start_time = self.time_to_seconds(parts[0])
            end_time = self.time_to_seconds(parts[1])
            speaker = parts[2]
            
            if speaker not in self.speakers:
                self.speakers[speaker] = []
            self.speakers[speaker].append((start_time, end_time))

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

    def train(self, audio_path, script_path):
        self.parse_script(script_path)
        train_data, _ = self.train_test_split()
        
        for speaker, segments in train_data.items():
            all_features = []
            for start, end in segments:
                xvector = self.extract_xvector(audio_path, start, end)
                q_features = self.process_qcnn(xvector)
                all_features.append(q_features)
            
            features = np.vstack(all_features)
            
            model = hmm.GaussianHMM(
                n_components=min(self.n_hmm_components, len(all_features)),
                covariance_type="diag", 
                n_iter=100
            )
            model.fit(features)
            self.hmm_models[speaker] = model
            print(f"Trained HMM model for {speaker} with {len(features)} samples")

    def predict(self, test_audio_path, start_time, end_time):
        test_xvector = self.extract_xvector(test_audio_path, start_time, end_time)
        test_features = self.process_qcnn(test_xvector)
    
        print(f"Test feature shape before reshape: {test_features.shape}")
    
        expected_frames, expected_features = next(iter(self.hmm_models.values())).means_.shape
        if len(test_features.shape) == 1:
            test_features = test_features.reshape(1, -1)
        if test_features.shape[1] != expected_features:
            print(f"⚠️ Feature dimension mismatch: {test_features.shape[1]} vs {expected_features}")
            return None, {}
    
        # Adjust frame count for HMM
        if test_features.shape[0] > expected_frames:
            test_features = test_features[:expected_frames, :]
        elif test_features.shape[0] < expected_frames:
            padding = np.zeros((expected_frames - test_features.shape[0], expected_features))
            test_features = np.vstack((test_features, padding))
    
        print(f"Reshaped test feature shape: {test_features.shape}")
    
        scores = {}
        for speaker, model in self.hmm_models.items():
            try:
                score = model.score(test_features)
                scores[speaker] = score
            except ValueError as e:
                print(f"Error scoring {speaker}: {e}")
    
        if not scores:
            return None, {}
    
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
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score (Macro-Averaged): {f1:.2f}")
        
        return precision, recall, f1

def main():
    si = SpeakerIdentification()
    
    audio_file = "meeting_1/raw.WAV"
    script_file = "meeting_1/script.txt"
    si.train(audio_file, script_file)
    
    si.evaluate(audio_file)

if __name__ == "__main__":
    main()
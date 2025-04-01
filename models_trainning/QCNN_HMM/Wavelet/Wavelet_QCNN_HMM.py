import numpy as np
import librosa
import pennylane as qml
from hmmlearn import hmm
import os
from scipy.io import wavfile
import warnings
from sklearn.model_selection import train_test_split
import pywt
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')

class SpeakerIdentification:
    def __init__(self, n_wavelet_features=13, n_qubits=4, n_hmm_components=5, wavelet='db1', max_level=3):
        self.n_wavelet_features = n_wavelet_features
        self.n_qubits = n_qubits
        self.n_hmm_components = n_hmm_components
        self.wavelet = wavelet
        self.max_level = max_level
        self.speakers = {}
        self.hmm_models = {}
        
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

    def extract_wavelet_features(self, audio_path, start_time, end_time):
        y, sr = librosa.load(audio_path, sr=None)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        audio_segment = y[start_sample:end_sample]
        
        # Pad the signal if it's too short for the wavelet transform
        min_length = 2**self.max_level
        if len(audio_segment) < min_length:
            audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)), 'constant')
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(audio_segment, self.wavelet, level=self.max_level)
        
        # Extract features from wavelet coefficients
        features = []
        for coeff in coeffs:
            features.append(np.mean(coeff))
            features.append(np.std(coeff))
            features.append(np.max(coeff))
            features.append(np.min(coeff))
        
        # Select the first n_wavelet_features or pad if needed
        if len(features) > self.n_wavelet_features:
            features = features[:self.n_wavelet_features]
        else:
            features = np.pad(features, (0, self.n_wavelet_features - len(features)), 'constant')
        
        return np.array(features).reshape(1, -1)  # Return as 2D array for consistency

    def process_qcnn(self, wavelet_features):
        # Since wavelet features are already extracted per segment (not per frame like MFCC),
        # we process them directly
        if len(wavelet_features) < self.n_qubits:
            wavelet_features = np.pad(wavelet_features, (0, self.n_qubits - len(wavelet_features)))
        else:
            wavelet_features = wavelet_features[:self.n_qubits]
        
        q_features = self.qcnn(wavelet_features, self.weights)
        return np.array(q_features).reshape(1, -1)  # Return as 2D array

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
            if len(segments) > 1:  # Ensure there are enough segments to split
                train_segments, test_segments = train_test_split(segments, test_size=0.2)
                train_data[speaker] = train_segments
                test_data[speaker] = test_segments
        
        return train_data, test_data

    def train(self, audio_path, script_path):
        self.parse_script(script_path)
        
        # Filter out speakers with too few segments
        min_segments = 2
        train_data = {sp: seg for sp, seg in self.train_test_split()[0].items() 
                     if len(seg) >= min_segments}
        
        for speaker, segments in train_data.items():
            all_features = []
            for start, end in segments:
                wavelet_features = self.extract_wavelet_features(audio_path, start, end)
                # Process each feature vector in the window
                for feature_vec in wavelet_features:
                    q_features = self.process_qcnn(feature_vec)
                    all_features.append(q_features)
            
            features = np.vstack(all_features)
            
            # Ensure we have enough samples
            if len(features) < self.n_hmm_components:
                print(f"Warning: Not enough samples ({len(features)}) for speaker {speaker}")
                
            model = hmm.GaussianHMM(n_components=min(self.n_hmm_components, len(features)//2), 
                              covariance_type="diag", 
                              n_iter=100)
            model.fit(features)
            self.hmm_models[speaker] = model
            print(f"Trained HMM model for {speaker} with {len(features)} samples")

    def predict(self, test_audio_path, start_time, end_time):
        test_wavelet = self.extract_wavelet_features(test_audio_path, start_time, end_time)
        test_features = self.process_qcnn(test_wavelet[0])  # Process single feature vector
        
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
        
        # Calculate precision, recall, and F1-score (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score (Macro-Averaged): {f1:.2f}")
        
        return precision, recall, f1
        
def main():
    si = SpeakerIdentification(n_wavelet_features=13, wavelet='db4', max_level=4)
        
    audio_file = os.path.join("..", "..", "train_data", "meeting_1", "raw.wav")
    script_file = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")
    si.train(audio_file, script_file)
    
    si.evaluate(audio_file)

if __name__ == "__main__":
    main()
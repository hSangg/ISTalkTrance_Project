import warnings

import librosa
import numpy as np
import pennylane as qml
from hmmlearn import hmm
from sklearn.metrics import precision_recall_fscore_support
import os
import pickle
import json

warnings.filterwarnings('ignore')

class SpeakerIdentification:
    def __init__(self, n_mfcc=13, n_qubits=4, n_hmm_components=5):
        self.n_mfcc = n_mfcc
        self.n_qubits = n_qubits
        self.n_hmm_components = n_hmm_components
        self.speakers = {}
        self.hmm_models = {}
        self.save_dir= "mfcc_qcnn_hmm_single"
        
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

    def extract_mfcc(self, audio_path, start_time, end_time):
        y, sr = librosa.load(audio_path, sr=None)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        audio_segment = y[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=self.n_mfcc)
        return mfcc.T

    def process_qcnn(self, mfcc_features, step=5):
        quantum_features = []
        for i in range(0, len(mfcc_features), step):  # Chỉ lấy 1 frame mỗi 5 frame
            frame = mfcc_features[i]
            if len(frame) < self.n_qubits:
                frame = np.pad(frame, (0, self.n_qubits - len(frame)))
            else:
                frame = frame[:self.n_qubits]
            
            q_features = self.qcnn(frame, self.weights)
            quantum_features.append(q_features)
        
        return np.array(quantum_features)


    def parse_script(self, script_path):
        speakers_local = {}  # Chỉ lưu speaker cho từng script.txt riêng biệt
        
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
            if len(segments) > 1:  # Đảm bảo có đủ segment để chia
                train_segments, test_segments = train_test_split(segments, test_size=0.2)
                train_data[speaker] = train_segments
                test_data[speaker] = test_segments
        
        return train_data, test_data

    def train(self):
        train_data, _ = self.train_test_split()
        
        print(f"Starting training... {len(train_data)} speakers to process")
        os.makedirs(self.save_dir, exist_ok=True)  # Đảm bảo thư mục lưu model tồn tại
    
        for speaker, segments in train_data.items():
            print(f"\n🔹 Speaker: {speaker}, Expected Segments: {len(segments)}")
            all_features = []
            
            for audio_path, start, end in segments:
                mfcc = self.extract_mfcc(audio_path, start, end)
                q_features = self.process_qcnn(mfcc)
                all_features.append(q_features)
            
            features = np.vstack(all_features)
            
            print(f"Training HMM for {speaker} with {features.shape} data points...")
            
            model = hmm.GaussianHMM(n_components=self.n_hmm_components, 
                                    covariance_type="diag", 
                                    n_iter=100, verbose=False)
            model.fit(features)
            self.hmm_models[speaker] = model
            
            # Lưu mô hình HMM ngay sau khi train xong
            model_path = os.path.join(self.save_dir, f"{speaker}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✅ Saved HMM model for {speaker} at {model_path}")
    
            # Lưu trọng số QCNN ngay sau khi train speaker này
            weights_path = os.path.join(self.save_dir, "qcnn_weights.pkl")
            with open(weights_path, "wb") as f:
                pickle.dump(self.weights, f)
            print(f"💾 Updated QCNN weights at {weights_path}")
    
            # Cập nhật danh sách speakers
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
            print(f"🔄 Updated speakers list in {speakers_list_path}")





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
        
        # Calculate precision, recall, and F1-score (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score (Macro-Averaged): {f1:.2f}")
        
        return precision, recall, f1


        
def train_all_datasets(base_folder):
    si = SpeakerIdentification()
    
    datasets = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    # Gộp toàn bộ dữ liệu từ nhiều dataset
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


    # Train trên dữ liệu gộp từ tất cả dataset
    print("\nTraining on all datasets combined...")

    si.speakers = all_segments  # Cập nhật lại danh sách speaker với dữ liệu gộp
    si.train()  # Không cần truyền file vì dữ liệu đã được load

    return si  # Trả về mô hình đã train để test

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
        
        speakers_data = si.parse_script(script_file)  # Lấy dữ liệu từ script.txt của dataset hiện tại
        
        for speaker, segments in speakers_data.items():  # Duyệt qua speaker trong dataset hiện tại
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
    
    # Lưu các mô hình HMM
    for speaker, model in si.hmm_models.items():
        model_path = os.path.join(save_dir, f"{speaker}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    # Lưu trọng số QCNN
    weights_path = os.path.join(save_dir, "qcnn_weights.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump(si.weights, f)
    
    # Lưu danh sách speakers
    speakers_path = os.path.join(save_dir, "speakers.json")
    with open(speakers_path, "w") as f:
        json.dump(list(si.hmm_models.keys()), f)
    
    print(f"✅ Models saved in {save_dir}")

def load_model(load_dir="mfcc_qcnn_hmm_models"):
    si = SpeakerIdentification()
    
    # Load danh sách speakers
    speakers_path = os.path.join(load_dir, "speakers.json")
    with open(speakers_path, "r") as f:
        speakers = json.load(f)
    
    # Load các mô hình HMM
    for speaker in speakers:
        model_path = os.path.join(load_dir, f"{speaker}.pkl")
        with open(model_path, "rb") as f:
            si.hmm_models[speaker] = pickle.load(f)
    
    # Load trọng số QCNN
    weights_path = os.path.join(load_dir, "qcnn_weights.pkl")
    with open(weights_path, "rb") as f:
        si.weights = pickle.load(f)
    
    print(f"✅ Models loaded from {load_dir}")
    return si
if __name__ == "__main__":
    train_voice_folder = "train_voice"
    si_model = train_all_datasets(train_voice_folder)
    save_model(si_model)
    evaluate_all_datasets(si_model, train_voice_folder)

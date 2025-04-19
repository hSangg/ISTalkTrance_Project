import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from speechbrain.pretrained import SpeakerRecognition
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.model_selection import KFold
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)

N_QUBITS = 7
N_LAYERS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 10
SEGMENT_DURATION = 3.0
SAMPLE_RATE = 16000
N_FOLDS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dev = qml.device("lightning.gpu", wires=N_QUBITS)
@qml.qnode(dev, interface="torch", diff_method="best")
def quantum_circuit(inputs, weights):
    # Ensure inputs have correct dimensions
    assert inputs.shape[1] == N_QUBITS, f"Expected input shape (batch_size, {N_QUBITS}), got {inputs.shape}"

    for i in range(N_QUBITS):
        qml.RY(inputs[:, i], wires=i)  # Run through all batches

    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l][i][0], wires=i)
            qml.RZ(weights[l][i][1], wires=i)
        
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        if N_QUBITS > 1:
            qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class DvectorCompressor(nn.Module):
    def __init__(self, input_dim=192, output_dim=N_QUBITS):  # ECAPA-TDNN embedding size is 192
        super(DvectorCompressor, self).__init__()
        self.compression = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.compression(x)

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super(QuantumNeuralNetwork, self).__init__()
        
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            single_result = self.qlayer(x[i:i+1])
            results.append(single_result)
            
        return torch.cat(results, dim=0)

class QDVectorCNN(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=2, dvector_dim=192):  # ECAPA-TDNN embedding size
        super(QDVectorCNN, self).__init__()
        
        self.compressor = DvectorCompressor(input_dim=dvector_dim, output_dim=N_QUBITS)
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.fc_input_size = self._get_conv_output_size(n_qubits)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )
    
    def _get_conv_output_size(self, n_qubits):
        dummy_input = torch.zeros(1, 1, n_qubits)
        dummy_output = self.conv_layer(dummy_input)
        return dummy_output.view(1, -1).shape[1]
        
    def forward(self, x):        
        x = self.compressor(x)
        x = self.qnn(x)
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def evaluate_model(model, test_loader, label_encoder):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predictions = torch.max(outputs.data, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    
    metrics = {
        'precision_macro': precision_score(labels_np, preds_np, average='macro'),
        'precision_weighted': precision_score(labels_np, preds_np, average='weighted'),
        'recall_macro': recall_score(labels_np, preds_np, average='macro'),
        'recall_weighted': recall_score(labels_np, preds_np, average='weighted'),
        'f1_macro': f1_score(labels_np, preds_np, average='macro'),
        'f1_weighted': f1_score(labels_np, preds_np, average='weighted'),
        'accuracy': accuracy_score(labels_np, preds_np)
    }
    
    return metrics

def extract_dvectors(audio_segments, dvector_model):
    dvectors = []
    
    for segment in audio_segments:
        with torch.no_grad():
            embedding = dvector_model.encode_batch(torch.tensor(segment).unsqueeze(0))
            dvectors.append(embedding.squeeze().cpu().numpy())
    
    return np.array(dvectors)

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_script(script_file):
    segments = []
    try:
        with open(script_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = time_to_seconds(parts[0])
                    end_time = time_to_seconds(parts[1])
                    label = parts[2]
                    segments.append((start_time, end_time, label))
    except Exception as e:
        print(f"Error parsing script file {script_file}: {e}")
        raise
        
    return segments

def extract_audio_segments(audio_file, script_file, segment_duration=SEGMENT_DURATION):
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    segments = parse_script(script_file)
    
    dataset = []
    for start_time, end_time, label in segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        segment_audio = y[start_sample:end_sample]
        
        if len(segment_audio) < 1:
            continue
        
        samples_per_segment = int(segment_duration * sr)
        
        for i in range(0, len(segment_audio), samples_per_segment):
            segment = segment_audio[i:i+samples_per_segment]
            
            if len(segment) < samples_per_segment // 2:
                continue
                
            if len(segment) < samples_per_segment:
                padded = np.zeros(samples_per_segment)
                padded[:len(segment)] = segment
                segment = padded
            elif len(segment) > samples_per_segment:
                segment = segment[:samples_per_segment]
            
            dataset.append((segment, label))
    
    return dataset

def train_speaker_recognition_system():
    print("Loading SpeechBrain pretrained ECAPA-TDNN model...")
    dvector_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    train_voice_dir = "../../train_voice"
    
    speaker_segments = defaultdict(list)
    all_segments = []
    all_labels = []
    
    for meeting_dir in os.listdir(train_voice_dir):
        meeting_path = os.path.join(train_voice_dir, meeting_dir)
        
        if not os.path.isdir(meeting_path):
            continue
            
        audio_file = None
        script_file = os.path.join(meeting_path, "script.txt")

        for ext in ['.wav', '.WAV']:
            if os.path.exists(os.path.join(meeting_path, f"raw{ext}")):
                audio_file = os.path.join(meeting_path, f"raw{ext}")
                break

        if not audio_file or not os.path.exists(script_file):
            print(f"Skipping {meeting_dir}: Missing audio or script file")
            continue
            
        print(f"Processing {meeting_dir}...")
        
        dataset = extract_audio_segments(audio_file, script_file)
        
        for segment, label in dataset:
            speaker_segments[label].append(segment)
            all_segments.append(segment)
            all_labels.append(label)
    
    print("Extracting D-Vectors using SpeechBrain ECAPA-TDNN model...")
    dvectors = extract_dvectors(all_segments, dvector_model)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    n_classes = len(label_encoder.classes_)
    print(f"Found {n_classes} speaker classes: {label_encoder.classes_}")
    
    speaker_dvectors = defaultdict(list)
    speaker_encoded_labels = defaultdict(list)
    speaker_indices = defaultdict(list)
    
    for i, (dvector, label) in enumerate(zip(dvectors, all_labels)):
        speaker_dvectors[label].append(dvector)
        speaker_encoded_labels[label].append(label_encoder.transform([label])[0])
        speaker_indices[label].append(i)
    
    print("\nSegment counts per speaker:")
    for speaker, segments in speaker_segments.items():
        print(f"{speaker}: {len(segments)} segments")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    speaker_fold_indices = {}
    for speaker in speaker_indices:
        indices = np.array(speaker_indices[speaker])
        speaker_fold_indices[speaker] = list(kf.split(indices))
    
    fold_metrics = []
    
    dvectors_tensor = torch.tensor(dvectors, dtype=torch.float32)
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    
    with open('DVector_QCNN.txt', 'w') as f:
        f.write("DVector QCNN Speaker Recognition Results\n")
        f.write("======================================\n\n")
        
    for fold in range(N_FOLDS):
        print(f"\n=== Training fold {fold+1}/{N_FOLDS} ===")
        
        train_indices = []
        test_indices = []
        
        for speaker, fold_indices in speaker_fold_indices.items():
            speaker_indices_array = np.array(speaker_indices[speaker])
            
            if fold < len(fold_indices):
                train_idx, test_idx = fold_indices[fold]
                train_indices.extend(speaker_indices_array[train_idx])
                test_indices.extend(speaker_indices_array[test_idx])
        
        X_train = dvectors_tensor[train_indices]
        y_train = labels_tensor[train_indices]
        X_test = dvectors_tensor[test_indices]
        y_test = labels_tensor[test_indices]
        
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        train_dataset = SpeakerDataset(X_train, y_train)
        test_dataset = SpeakerDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        
        dvector_dim = dvectors_tensor.shape[1]  # Typically 192 for ECAPA-TDNN
        
        qcnn_model = QDVectorCNN(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_classes=n_classes,
            dvector_dim=dvector_dim
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(qcnn_model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        for epoch in range(EPOCHS):
            qcnn_model.train()
            running_loss = 0.0
            
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                if batch_features.size(0) == 1:
                    print("Skipping batch size 1")
                    continue
                
                optimizer.zero_grad()
                
                outputs = qcnn_model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            
            qcnn_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = qcnn_model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_accuracy = 100 * correct / total if total > 0 else 0
            
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        metrics = evaluate_model(qcnn_model, test_loader, label_encoder)
        metrics['fold'] = fold + 1
        fold_metrics.append(metrics)
        
        print(f"\nFold {fold+1} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        torch.save(qcnn_model.state_dict(), f'qdvector_speaker_recognition_fold{fold+1}.pth')
        
        with open('DVector_QCNN.txt', 'a') as f:
            f.write(f"Fold {metrics['fold']} Results:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n\n")
    
    with open('DVector_QCNN.txt', 'a') as f:
        avg_accuracy = sum(m['accuracy'] for m in fold_metrics) / len(fold_metrics)
        avg_precision = sum(m['precision_macro'] for m in fold_metrics) / len(fold_metrics)
        avg_recall = sum(m['recall_macro'] for m in fold_metrics) / len(fold_metrics)
        avg_f1_macro = sum(m['f1_macro'] for m in fold_metrics) / len(fold_metrics)
        avg_f1_weighted = sum(m['f1_weighted'] for m in fold_metrics) / len(fold_metrics)
        
        f.write("Average Results Across All Folds:\n")
        f.write(f"Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Precision (Macro): {avg_precision:.4f}\n")
        f.write(f"Recall (Macro): {avg_recall:.4f}\n")
        f.write(f"F1-Score (Macro): {avg_f1_macro:.4f}\n")
        f.write(f"F1-Score (Weighted): {avg_f1_weighted:.4f}\n")
    
    print("\nTraining complete! Results saved to DVector_QCNN.txt")

if __name__ == "__main__":
    train_speaker_recognition_system()
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def find_wav_txt_pairs(root_dir):
    """Tìm tất cả các cặp file raw.wav và script.txt trong các thư mục con"""
    pairs = []
    # Lấy danh sách thư mục con trực tiếp
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print("subdirs: ", subdirs)
    
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        files = os.listdir(subdir_path)
        
        # Kiểm tra xem trong thư mục có cả raw.wav và script.txt không
        if 'raw.wav' in files and 'script.txt' in files:
            pairs.append({
                'wav_path': os.path.join(subdir_path, 'raw.wav'),
                'txt_path': os.path.join(subdir_path, 'script.txt')
            })
            print(f"Tìm thấy cặp file trong thư mục {subdir}")
    
    print(f"Tổng số cặp file tìm thấy: {len(pairs)}")
    return pairs

def load_labels(script_path):
    """Đọc và xử lý file script.txt"""
    labels = []
    timestamps = []
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:  # Added encoding
            print(f"Reading script file: {script_path}")
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        print(f"Warning: Line {line_num} has incorrect format: {line.strip()}")
                        continue
                    start_time, end_time, speaker = parts
                    print(f"Processing line {line_num}: {start_time} - {end_time} - {speaker}")
                    labels.append(speaker)
                    timestamps.append((start_time, end_time))
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        print(f"Successfully loaded {len(labels)} entries from script file")
        return timestamps, labels
    except Exception as e:
        print(f"Error opening script file: {str(e)}")
        raise
    
def time_to_seconds(time_str):
    """Chuyển đổi timestamp sang seconds"""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s


def extract_mfcc(audio_path, start_time, end_time, sr=16000):
    """
    Trích xuất MFCC features từ đoạn audio với delta features
    
    Args:
        audio_path: Đường dẫn file audio
        start_time: Thời gian bắt đầu (format HH:MM:SS)
        end_time: Thời gian kết thúc (format HH:MM:SS)
        sr: Sample rate (default: 16000)
    
    Returns:
        numpy.ndarray: MFCC features kết hợp với delta features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Convert timestamps to samples
        start_sample = int(time_to_seconds(start_time) * sr)
        end_sample = int(time_to_seconds(end_time) * sr)
        
        # Extract audio segment
        audio_segment = y[start_sample:end_sample]
        
        # Extract base MFCC features
        mfcc_features = librosa.feature.mfcc(
            y=audio_segment, 
            sr=sr,
            n_mfcc=20  # Keeping 20 MFCCs as in original
        )
        
        # Calculate delta features
        mfcc_delta = librosa.feature.delta(mfcc_features)
        mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
        
        # Combine all features
        combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
        
        # Transpose to get time as first dimension
        combined_mfcc = combined_mfcc.T
        
        # Pad or truncate to fixed length (100 frames as in original)
        if combined_mfcc.shape[0] < 100:
            combined_mfcc = np.pad(
                combined_mfcc,
                ((0, 100 - combined_mfcc.shape[0]), (0, 0))
            )
        else:
            combined_mfcc = combined_mfcc[:100, :]
        
        return combined_mfcc
        
    except Exception as e:
        raise Exception(f"Error extracting MFCC features: {str(e)}")


class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]

class SpeakerRecognitionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpeakerRecognitionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

def process_all_data(root_dir):
    """Xử lý tất cả dữ liệu từ các thư mục con"""
    pairs = find_wav_txt_pairs(root_dir)
    print("Found file pairs:", pairs)
    all_features = []
    all_labels = []
    
    print(f"Processing {len(pairs)} wav/txt pairs")
    
    for pair in tqdm(pairs, desc="Processing files"):
        print(f"\nProcessing pair: {pair}")
        try:
            # Verify files exist
            if not os.path.exists(pair['wav_path']):
                print(f"WAV file not found: {pair['wav_path']}")
                continue
            if not os.path.exists(pair['txt_path']):
                print(f"TXT file not found: {pair['txt_path']}")
                continue
                
            # Đọc timestamps và labels từ file txt
            print(f"Loading labels from: {pair['txt_path']}")
            timestamps, labels = load_labels(pair['txt_path'])
            print(f"Found {len(timestamps)} segments in script")
            
            # Check audio file
            try:
                y, sr = librosa.load(pair['wav_path'], sr=16000, duration=10)  # Load first 10s to test
                print(f"Successfully loaded audio file: {pair['wav_path']}")
                print(f"Audio length: {librosa.get_duration(y=y, sr=sr):.2f}s")
            except Exception as e:
                print(f"Error loading audio file {pair['wav_path']}: {str(e)}")
                continue
            
            # Xử lý từng đoạn trong file
            for i, ((start_time, end_time), speaker) in enumerate(zip(timestamps, labels)):
                try:
                    print(f"\nProcessing segment {i+1}/{len(timestamps)}")
                    print(f"Time: {start_time} - {end_time}, Speaker: {speaker}")
                    
                    # Trích xuất MFCC features
                    mfcc = extract_mfcc(pair['wav_path'], start_time, end_time)
                    print(f"Extracted MFCC shape: {mfcc.shape}")
                    
                    all_features.append(mfcc)
                    all_labels.append(speaker)
                    print(f"Successfully processed segment {i+1}")
                except Exception as e:
                    print(f"Error processing segment {i+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing file pair: {str(e)}")
            continue
    
    if not all_features:
        raise Exception("No data was successfully processed!")
    
    print(f"\nProcessing complete:")
    print(f"Total features extracted: {len(all_features)}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Feature shape: {all_features[0].shape}")
        
    return np.array(all_features), all_labels

def train_speaker_recognition(root_dir, epochs=100):
    # Xử lý tất cả dữ liệu
    print("Bắt đầu xử lý dữ liệu...")
    features, labels = process_all_data(root_dir)
    
    # Encode labels
    le = LabelEncoder()

    labels_encoded = le.fit_transform(labels)
    
    print(f"Tổng số mẫu: {len(features)}")
    print(f"Số lượng người nói: {len(le.classes_)}")
    print("Danh sách người nói:", le.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = SpeakerDataset(X_train, y_train)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = 60  # number of MFCC features
    hidden_size = 128
    num_layers = 2
    num_classes = len(le.classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeakerRecognitionRNN(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Bắt đầu training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    
    # Lưu model và label encoder
    print("Lưu model và label encoder...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': le,
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes
        }
    }, "speaker_recognition_model.pth")
    
    return model, le

if __name__ == "__main__":
    root_dir = "train_voice"
    model, label_encoder = train_speaker_recognition(root_dir)
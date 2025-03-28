import torch
import librosa
import numpy as np
from torch import nn

def load_model(model_path):
    """Load model đã train"""
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Khởi tạo model với config đã lưu
    config = checkpoint['model_config']
    model = SpeakerRecognitionRNN(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load label encoder
    label_encoder = checkpoint['label_encoder']
    
    return model, label_encoder

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

def extract_mfcc(audio_path, start_time, end_time, sr=16000, n_mfcc=20):
    """Trích xuất MFCC features từ đoạn audio"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    start_sample = int(time_to_seconds(start_time) * sr)
    end_sample = int(time_to_seconds(end_time) * sr)
    
    audio_segment = y[start_sample:end_sample]
    
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc)
    
    # Change padding/truncation to match model's expected input size of 60
    if mfcc.shape[1] < 60:
        mfcc = np.pad(mfcc, ((0, 0), (0, 60 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :60]
    
    return mfcc

def time_to_seconds(time_str):
    """Chuyển đổi timestamp sang seconds"""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def predict_speaker(model, label_encoder, audio_path, start_time, end_time):
    """Dự đoán người nói cho một đoạn audio"""
    # Đưa model về chế độ evaluation
    model.eval()
    
    # Trích xuất features
    features = extract_mfcc(audio_path, start_time, end_time)
    
    # Chuyển features thành tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Thêm batch dimension
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        # Lấy tên người nói từ label encoder
        speaker = label_encoder.inverse_transform(predicted.numpy())[0]
        
        # Tính probability
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted[0]].item()
        
    return speaker, confidence

def predict_from_file(model_path, audio_path, script_path):
    """Dự đoán người nói cho toàn bộ file audio dựa trên script"""
    # Load model
    model, label_encoder = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Đọc script file
    results = []
    with open(script_path, 'r') as f:
        for line in f:
            start_time, end_time, _ = line.strip().split()
            
            # Dự đoán người nói
            predicted_speaker, confidence = predict_speaker(
                model, label_encoder, audio_path, start_time, end_time
            )
            
            results.append({
                'start_time': start_time,
                'end_time': end_time,
                'predicted_speaker': predicted_speaker,
                'confidence': confidence
            })
    
    return results

# Example usage
if __name__ == "__main__":
    # # Dự đoán cho một file cụ thể
    model_path = "speaker_recognition_model.pth"
    audio_path = "train_voice/vnoi_talkshow/raw.wav"
    # script_path = "path/to/test.txt"
    
    # # Dự đoán cho toàn bộ file
    # results = predict_from_file(model_path, audio_path, script_path)
    
    # # In kết quả
    # print("\nKết quả dự đoán:")
    # for result in results:
    #     print(f"Thời gian: {result['start_time']} - {result['end_time']}")
    #     print(f"Người nói: {result['predicted_speaker']}")
    #     print(f"Độ tin cậy: {result['confidence']*100:.2f}%\n")
    
    # Hoặc dự đoán cho một đoạn cụ thể
    model, label_encoder = load_model(model_path)
    speaker, confidence = predict_speaker(
        model, 
        label_encoder,
        audio_path,
        "0:29:48",  # start time
        "0:30:00"   # end time
    )
    print(f"Đoạn 0:00:00 - 0:00:30:")
    print(f"Người nói: {speaker}")
    print(f"Độ tin cậy: {confidence*100:.2f}%")

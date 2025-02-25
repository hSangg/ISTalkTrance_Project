import numpy as np
import torchaudio
import joblib
import pywt
from hmmlearn import hmm
import os
import torch

def time_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s

def load_script(script_path):
    segments, speakers = [], []
    with open(script_path, "r", encoding="utf-8") as file:
        for line in file:
            start, end, speaker = line.strip().split()
            segments.append((time_to_seconds(start), time_to_seconds(end)))
            speakers.append(speaker)
    return segments, speakers

def extract_wavelet_features(audio_path, segments_speakers, wavelet='db4', level=1):
    """
    Trích xuất đặc trưng bằng biến đổi wavelet thay vì X-vector
    
    Parameters:
    - audio_path: Đường dẫn đến file âm thanh
    - segments_speakers: Danh sách các cặp ((start, end), speaker)
    - wavelet: Loại wavelet sử dụng (mặc định: 'db4')
    - level: Số cấp độ phân tích wavelet (mặc định: 5)
    
    Returns:
    - Dictionary chứa đặc trưng wavelet cho mỗi người nói
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)  # Chuyển stereo thành mono
    
    waveform = waveform.numpy().squeeze()
    
    print(f"⚡ Sample rate: {sample_rate}, Số mẫu: {waveform.shape}")

    feature_dict = {}
    
    # Hàm an toàn để tính các thống kê
    def safe_stat(func, arr, default=0.0):
        if len(arr) == 0:
            return default
        return func(arr)
    
    for (start, end), speaker in segments_speakers:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        print("start sample: ", start_sample)
        print("end sample: ", end_sample)
        segment_waveform = waveform[start_sample:end_sample]
        print("segment_waveform len():: ", len(segment_waveform))
        # Đảm bảo đoạn âm thanh đủ dài cho phân tích wavelet
        if len(segment_waveform) < 2**level:
            print(f"⚠️ Đoạn âm thanh quá ngắn cho người nói {speaker}, đang bỏ qua")
            continue
        
        if len(segment_waveform) == 0:
            print(f"⚠️ Cảnh báo: Đoạn {start}-{end} của {speaker} bị rỗng!")
            continue  # Bỏ qua đoạn này

        # Thực hiện biến đổi wavelet đa phân giải
        coeffs = pywt.wavedec(segment_waveform, wavelet, level=min(level, pywt.dwt_max_level(len(segment_waveform), pywt.Wavelet(wavelet).dec_len)))
        
        # Tạo vector đặc trưng từ các hệ số wavelet
        features = []
        for i, coeff in enumerate(coeffs):
            # Tính các đặc trưng thống kê từ mỗi mức wavelet (với xử lý mảng rỗng)
            features.extend([
                safe_stat(np.mean, coeff),        # Giá trị trung bình
                safe_stat(np.std, coeff, 1.0),    # Độ lệch chuẩn
                safe_stat(np.max, coeff),         # Giá trị lớn nhất
                safe_stat(np.min, coeff),         # Giá trị nhỏ nhất
                safe_stat(np.median, coeff),      # Giá trị trung vị
                safe_stat(lambda x: np.sum(x**2), coeff)  # Năng lượng
            ])
        
        # Chuyển thành mảng numpy
        features = np.array(features)
        
        # Thêm vào dictionary
        if speaker not in feature_dict:
            feature_dict[speaker] = []
        feature_dict[speaker].append(features)
    
    return feature_dict

def train_hmm_for_speaker(speaker, features):
    if not features:
        print(f"⚠️ Không có đặc trưng để huấn luyện cho người nói {speaker}")
        return
    
    features = np.vstack(features)  # Chuyển danh sách thành mảng numpy
    
    # Chuẩn hóa đặc trưng (quan trọng cho wavelet)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Tránh chia cho 0 hoặc NaN
    std[std < 1e-10] = 1.0
    features = (features - mean) / std
    
    n_components = min(3, len(features))
    model = hmm.GaussianHMM(n_components, covariance_type="diag", n_iter=100)
    model.fit(features)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{speaker}_model.pkl")
    # Lưu thêm thông số chuẩn hóa để sử dụng khi dự đoán
    joblib.dump((mean, std), f"models/{speaker}_norm_params.pkl")
    print(f"✅ Đã lưu mô hình HMM cho {speaker}")

def load_hmm_model(speaker):
    model_path = f"models/{speaker}_model.pkl"
    norm_params_path = f"models/{speaker}_norm_params.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(norm_params_path):
        print(f"⚠️ Không tìm thấy mô hình cho {speaker}")
        return None, None
    
    model = joblib.load(model_path)
    mean, std = joblib.load(norm_params_path)
    return model, (mean, std)

def predict_speaker(speaker, features):
    model, norm_params = load_hmm_model(speaker)
    if model is None or not features:
        return []
    
    features = np.vstack(features)
    mean, std = norm_params
    # Tránh chia cho 0 hoặc NaN trong chuẩn hóa
    std[std < 1e-10] = 1.0
    features = (features - mean) / std
    
    return model.predict(features)

# ======= Chạy thử nghiệm =======
audio_file = "train_voice/extraordinary_strategic/raw.WAV"
script_file = "train_voice/extraordinary_strategic/script.txt"

# Load script
segments, speakers = load_script(script_file)

# Trích xuất đặc trưng wavelet
feature_dict = extract_wavelet_features(audio_file, zip(segments, speakers))

# Huấn luyện mô hình HMM
for speaker, features in feature_dict.items():
    train_hmm_for_speaker(speaker, features)

# # Dự đoán (nếu cần)
# for speaker, features in feature_dict.items():
#     predictions = predict_speaker(speaker, features)
#     print(f"🔍 Dự đoán cho {speaker}: {predictions}")
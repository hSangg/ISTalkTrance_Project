import os
from collections import Counter

# Đường dẫn thư mục gốc chứa các subfolder
root_dir = 'train_voice'

# Tập hợp tất cả speaker xuất hiện
all_speakers = []

# Bước 1: Duyệt qua tất cả script.txt để đếm số lần xuất hiện của speaker
for subfolder in os.listdir(root_dir):
    script_path = os.path.join(root_dir, subfolder, 'script.txt')
    if os.path.isfile(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    speaker = parts[-1]
                    all_speakers.append(speaker)

# Đếm số lần xuất hiện
speaker_counts = Counter(all_speakers)

# Tìm speaker xuất hiện ít hơn 4 lần
rare_speakers = {speaker for speaker, count in speaker_counts.items() if count < 10}

print("⚠️ Những speaker xuất hiện ít hơn 4 lần:")
for speaker in rare_speakers:
    print(f"  - {speaker}: {speaker_counts[speaker]} lần")

# Bước 2: Lọc lại script.txt và xoá script_predicted.txt nếu có
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    script_path = os.path.join(subfolder_path, 'script.txt')
    predicted_path = os.path.join(subfolder_path, 'script_predicted.txt')

    # Xoá script_predicted.txt nếu tồn tại
    if os.path.isfile(predicted_path):
        os.remove(predicted_path)
        print(f"🗑️ Đã xóa {predicted_path}")

    # Lọc lại script.txt
    if os.path.isfile(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue  # Bỏ dòng rỗng

            parts = line.split()
            if len(parts) < 3:
                continue  # Bỏ dòng thiếu start, end, speaker

            start, end = parts[0], parts[1]
            speaker = parts[-1]

            if start == end:
                continue  # Bỏ dòng có start == end

            if speaker in rare_speakers:
                continue

            filtered_lines.append(line + '\n')  # Thêm lại dòng hợp lệ

        with open(script_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)

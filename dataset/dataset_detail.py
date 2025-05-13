import os
from collections import defaultdict, Counter
from datetime import datetime
import math

def parse_timestamp(timestamp: str) -> int:
    """Chuyển chuỗi thời gian 'HH:MM:SS' thành giây"""
    h, m, s = map(int, timestamp.strip().split(":"))
    return h * 3600 + m * 60 + s

def std_dev(data):
    if len(data) < 2:
        return 0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def analyze_voice_dataset(dataset_path: str):
    speaker_stats = defaultdict(lambda: {
        'total_time': 0,
        'segment_count': 0,
        'durations': []
    })

    total_segments = 0
    total_audio_files = 0
    txt_file_count = 0
    speaker_sequence = []
    all_durations = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_file_count += 1
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 3:
                            continue
                        start, end, speaker = parts
                        try:
                            start_sec = parse_timestamp(start)
                            end_sec = parse_timestamp(end)
                            duration = end_sec - start_sec
                            if duration < 0:
                                continue

                            speaker_stats[speaker]['total_time'] += duration
                            speaker_stats[speaker]['segment_count'] += 1
                            speaker_stats[speaker]['durations'].append(duration)

                            speaker_sequence.append(speaker)
                            all_durations.append(duration)
                            total_segments += 1
                        except:
                            continue
            elif file.lower().endswith('.wav'):
                total_audio_files += 1

    # Tính số lần chuyển speaker
    speaker_turns = sum(1 for i in range(1, len(speaker_sequence)) if speaker_sequence[i] != speaker_sequence[i - 1])

    # In kết quả
    print("📊 Thống kê tập dữ liệu thoại:")
    print(f"- Tổng số speaker             : {len(speaker_stats)}")
    print(f"- Tổng số đoạn thoại          : {total_segments}")
    print(f"- Số lần chuyển speaker       : {speaker_turns}")
    print(f"- Tổng số file audio          : {total_audio_files}")
    print(f"- Tổng số file script (.txt)  : {txt_file_count}")
    print(f"- Tổng thời lượng toàn tập    : {sum(all_durations)} giây (~{sum(all_durations)//60} phút)")
    print(f"- Độ dài TB đoạn thoại        : {sum(all_durations)/len(all_durations):.2f} giây")
    print(f"- Độ lệch chuẩn độ dài         : {std_dev(all_durations):.2f} giây")

    print("\n🔍 Phân phối thời lượng theo speaker (%):")
    total_time = sum(all_durations)
    for speaker, stats in speaker_stats.items():
        ratio = stats['total_time'] / total_time * 100 if total_time > 0 else 0
        print(f"  - {speaker:10}: {ratio:.2f}%")

    print("\n🔎 Chi tiết theo speaker:")
    for speaker, stats in speaker_stats.items():
        avg_duration = stats['total_time'] / stats['segment_count'] if stats['segment_count'] else 0
        dev = std_dev(stats['durations'])
        print(f"\n👤 Speaker: {speaker}")
        print(f"  - Số đoạn thoại       : {stats['segment_count']}")
        print(f"  - Tổng thời lượng     : {stats['total_time']} giây (~{stats['total_time'] // 60} phút)")
        print(f"  - Trung bình đoạn     : {avg_duration:.2f} giây")
        print(f"  - Độ lệch chuẩn đoạn  : {dev:.2f} giây")
        
    min_duration = min(all_durations) if all_durations else 0
    max_duration = max(all_durations) if all_durations else 0
    sorted_durations = sorted(all_durations)
    median_duration = sorted_durations[len(sorted_durations)//2] if all_durations else 0
    mode_duration = Counter(all_durations).most_common(1)[0][0] if all_durations else 0

    # Phân phối độ dài
    short = sum(1 for d in all_durations if d < 3)
    mid = sum(1 for d in all_durations if 3 <= d < 5)
    long = sum(1 for d in all_durations if 5 <= d <= 10)
    very_long = sum(1 for d in all_durations if d > 10)

    print("\n📈 Thống kê mô tả độ dài đoạn thoại:")
    print(f"- Độ dài ngắn nhất             : {min_duration} giây")
    print(f"- Độ dài dài nhất              : {max_duration} giây")
    print(f"- Độ dài trung vị (median)     : {median_duration} giây")
    print(f"- Độ dài phổ biến nhất (mode)  : {mode_duration} giây")

    print("\n📊 Phân phối độ dài đoạn thoại:")
    print(f"- < 3 giây      : {short} đoạn ({100 * short / total_segments:.2f}%)")
    print(f"- 3–5 giây      : {mid} đoạn ({100 * mid / total_segments:.2f}%)")
    print(f"- 5–10 giây     : {long} đoạn ({100 * long / total_segments:.2f}%)")
    print(f"- > 10 giây     : {very_long} đoạn ({100 * very_long / total_segments:.2f}%)")


analyze_voice_dataset("test_voice")

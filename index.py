import json

def merge_same_speaker_data_segments(results):
    if not results:
        return []

    merged_results = []
    current_segment = {
        "start_time": results[0]["start_time"],
        "end_time": results[0]["end_time"],
        "speaker_data": results[0]["speaker_data"],
        "transcription": results[0]["transcription"]
    }

    for i in range(1, len(results)):
        segment = results[i]
        if segment["speaker_data"] == current_segment["speaker_data"]:
            # Gộp đoạn lại
            current_segment["end_time"] = segment["end_time"]
            current_segment["transcription"] += " " + segment["transcription"]
        else:
            # Đẩy đoạn cũ vào danh sách kết quả
            merged_results.append(current_segment)
            # Bắt đầu đoạn mới
            current_segment = {
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "speaker_data": segment["speaker_data"],
                "transcription": segment["transcription"]
            }

    # Thêm đoạn cuối cùng
    merged_results.append(current_segment)
    return merged_results



def export_results_to_json(results, json_file_path):
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Successfully exported results to {json_file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the JSON file: {e}")


json_file = "test_voice/vuiveuncut_conentinkiemtientrenmang/speech.json" #example

# Load the JSON file
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found at {json_file}")
    results = []  # Ensure results is initialized to an empty list
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {json_file}")
    results = []


# Apply the merge function
merged_results = merge_same_speaker_data_segments(results)

# Store the merged JSON back to the same file
export_results_to_json(merged_results, json_file)


# dialogue_text = "\n".join(
#         f'{entry["speaker_data"]}: {entry["transcription"]}' for entry in merge_same_speaker_data_segments(result)
#     )

# print(dialogue_text)
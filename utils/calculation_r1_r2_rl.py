import json
import os

from rouge_score import rouge_scorer

# Tạo RougeScorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

base_path = "test_voice"
overall_results = {"rouge1": [], "rouge2": [], "rougeL": []}

# Duyệt tất cả các thư mục con trong test_voice
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    true_file = os.path.join(folder_path, "true_summarization.json")
    test_file = os.path.join(folder_path, "test_summarization.json")

    # Bỏ qua nếu không tìm thấy file cần thiết
    if not os.path.isfile(true_file) or not os.path.isfile(test_file):
        print(f"Bỏ qua {folder_name} vì thiếu file.")
        continue

    with open(true_file, "r", encoding="utf-8") as f:
        true_data = json.load(f)

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    references = [item["transcription"] for item in true_data]
    predictions = [item["transcription"] for item in test_data]

    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        results["rouge1"].append(scores["rouge1"].fmeasure)
        results["rouge2"].append(scores["rouge2"].fmeasure)
        results["rougeL"].append(scores["rougeL"].fmeasure)

    # Tính trung bình từng chỉ số ROUGE cho thư mục hiện tại
    avg = {key: sum(values) / len(values) if values else 0 for key, values in results.items()}

    print(f"\n📁 Kết quả cho thư mục: {folder_name}")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    # Thêm vào tổng
    for key in overall_results:
        overall_results[key].extend(results[key])

# Tính điểm trung bình toàn bộ
print("\n🌟 Tổng kết OVERALL:")
for key in overall_results:
    all_scores = overall_results[key]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"{key}: {avg_score:.4f}")

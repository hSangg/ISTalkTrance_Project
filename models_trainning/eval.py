import os
from collections import defaultdict

from sklearn.metrics import accuracy_score, classification_report

SCRIPT_PREDICTED_TXT = "script_predicted.txt"


def parse_script_file(file_path):
    segments = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:  # Need at least start, end, and speaker
                    continue

                start_time, end_time, speaker = parts[0], parts[1], parts[2]
                segments.append((start_time, end_time, speaker))
        return segments
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def evaluate_subfolder(subfolder_path):
    true_script_path = os.path.join(subfolder_path, "script.txt")
    pred_script_path = os.path.join(subfolder_path, SCRIPT_PREDICTED_TXT)

    if not os.path.exists(true_script_path):
        print(f"Warning: Ground truth script not found in {subfolder_path}")
        return None

    if not os.path.exists(pred_script_path):
        print(f"Warning: Predicted script not found in {subfolder_path}")
        return None

    true_segments = parse_script_file(true_script_path)
    pred_segments = parse_script_file(pred_script_path)

    if not true_segments or not pred_segments:
        print(f"Warning: Empty segments in {subfolder_path}")
        return None

    if len(true_segments) != len(pred_segments):
        print(f"Warning: Segment count mismatch in {subfolder_path}. "
              f"True: {len(true_segments)}, Predicted: {len(pred_segments)}")
        min_length = min(len(true_segments), len(pred_segments))
        true_segments = true_segments[:min_length]
        pred_segments = pred_segments[:min_length]

    y_true = [segment[2] for segment in true_segments]
    y_pred = [segment[2] for segment in pred_segments]

    return {
        'folder': os.path.basename(subfolder_path),
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy_score(y_true, y_pred) if y_true else 0
    }


def evaluate_all_subfolders(test_voice_folder):
    all_results = []
    all_true = []
    all_pred = []

    for subfolder in os.listdir(test_voice_folder):
        subfolder_path = os.path.join(test_voice_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"Evaluating folder: {subfolder}")
        result = evaluate_subfolder(subfolder_path)

        if result:
            all_results.append(result)
            all_true.extend(result['y_true'])
            all_pred.extend(result['y_pred'])

    return all_results, all_true, all_pred

def generate_evaluation_report(results, all_true, all_pred):

    overall_accuracy = accuracy_score(all_true, all_pred)

    folder_accuracies = {result['folder']: result['accuracy'] for result in results}

    speaker_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for true, pred in zip(all_true, all_pred):
        speaker_stats[true]['total'] += 1
        if true == pred:
            speaker_stats[true]['correct'] += 1

    speaker_accuracies = {
        speaker: (stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0)
        for speaker, stats in speaker_stats.items()
    }

    report = classification_report(all_true, all_pred, output_dict=True)

    with open('test_voice/evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("SPEAKER RECOGNITION MODEL EVALUATION\n")
        f.write("==================================\n\n")

        f.write(
            f"Overall Accuracy: {overall_accuracy:.4f} ({sum(1 for t, p in zip(all_true, all_pred) if t == p)} correct out of {len(all_true)} total)\n\n")

        f.write("Per-Folder Accuracy:\n")
        f.write("-----------------\n")
        for folder, acc in sorted(folder_accuracies.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{folder}: {acc:.4f}\n")

        f.write("\nPer-Speaker Accuracy:\n")
        f.write("-------------------\n")
        for speaker, acc in sorted(speaker_accuracies.items(), key=lambda x: x[1], reverse=True):
            f.write(
                f"{speaker}: {acc:.2f}% ({speaker_stats[speaker]['correct']} correct out of {speaker_stats[speaker]['total']} total)\n")

        f.write("\nDetailed Classification Report:\n")
        f.write("-----------------------------\n")
        f.write(f"{'Speaker':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        for speaker in sorted(report.keys()):
            if speaker not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[speaker]
                f.write(
                    f"{speaker:<15} {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1-score']:.4f}      {metrics['support']:<10}\n")

    print("Evaluation report saved as 'evaluation_report.txt'")


if __name__ == "__main__":
    test_voice_folder = "test_voice_pho"

    print(f"Starting evaluation of all subfolders in {test_voice_folder}...")
    all_results, all_true_labels, all_pred_labels = evaluate_all_subfolders(test_voice_folder)

    if all_results:
        print(f"Processed {len(all_results)} valid subfolders.")
        generate_evaluation_report(all_results, all_true_labels, all_pred_labels)
        print("Evaluation complete!")
    else:
        print("No valid results found for evaluation.")
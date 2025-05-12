import os
import json

VIDEO_INPUT_DIR = "vinput"
VIDEO_OUTPUT_DIR = "voutput"
LABELS_FILE = "ground_truth.json"
CATEGORIES_FILE = "reference/categories.txt"
VALID_EXTS = (".mp4", ".mov", ".avi", ".mkv")

def load_categories(path):
    categories = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                idx, name = line.strip().split(':', 1)
                categories[idx.strip()] = name.strip().lower()
    return categories

def collect_all_videos(root_dir):
    videos = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                full_path = os.path.join(root, f)
                videos.append((f, full_path))
    return videos

def prompt_ground_truth(videos, categories):
    print("=== Ground Truth Collection ===")
    print("Category Options:")
    for k, v in categories.items():
        print(f" {k}: {v}")
    print()

    ground_truth = {}
    for filename, full_path in sorted(videos):
        while True:
            guess = input(f"What game is '{full_path}'? [0-3]: ").strip()
            if guess in categories:
                ground_truth[filename] = categories[guess]
                break
            else:
                print("Invalid input. Please enter one of:", ', '.join(categories.keys()))
    with open(LABELS_FILE, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\n[SAVED] Ground truth to {LABELS_FILE}")
    return ground_truth

def collect_predictions(output_dir):
    predictions = {}
    for category in os.listdir(output_dir):
        cat_path = os.path.join(output_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for f in os.listdir(cat_path):
            if f.lower().endswith(VALID_EXTS):
                predictions[f] = category.lower()
    return predictions

def evaluate(ground_truth, predictions):
    total = len(ground_truth)
    correct = 0
    mismatches = []

    for filename, true_label in ground_truth.items():
        predicted_label = predictions.get(filename)
        if predicted_label == true_label:
            correct += 1
        else:
            mismatches.append((filename, true_label, predicted_label))

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total, mismatches

if __name__ == "__main__":
    categories = load_categories(CATEGORIES_FILE)
    all_videos = collect_all_videos(VIDEO_INPUT_DIR)

    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            ground_truth = json.load(f)
        print(f"[LOADED] Existing ground truth from {LABELS_FILE}")
    else:
        ground_truth = prompt_ground_truth(all_videos, categories)

    predictions = collect_predictions(VIDEO_OUTPUT_DIR)
    accuracy, correct, total, mismatches = evaluate(ground_truth, predictions)

    print("\n=== Accuracy Report ===")
    print(f"Total videos: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    if mismatches:
        print("Mismatches:")
        for f, expected, predicted in mismatches:
            print(f" - {f}: expected '{expected}', got '{predicted or 'none'}'")

import os
import sys
import shutil
import time
import lmstudio as lms
import re
from collections import defaultdict
from tqdm import tqdm

threshold = 5

REFERENCE_FOLDER = os.path.join(os.getcwd(), 'reference')
INPUT_FOLDER = os.path.join(os.getcwd(), 'input')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
MODEL_ID = "minicpm-o-2_6"

def load_categories_and_references(ref_folder):
    categories_path = os.path.join(ref_folder, 'categories.txt')
    if not os.path.exists(categories_path):
        raise FileNotFoundError("Missing categories.txt in reference folder")

    category_map = {}
    with open(categories_path, 'r') as f:
        for line in f:
            if ':' in line:
                idx, name = line.strip().split(':', 1)
                category_map[str(int(idx.strip()))] = name.strip().lower()

    references = defaultdict(list)
    for file in os.listdir(ref_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            name = os.path.splitext(file)[0]
            parts = name.split('-')
            if parts[0].isdigit():
                category_index = str(int(parts[0]))
                if category_index in category_map:
                    category = category_map[category_index]
                    references[category].append(os.path.join(ref_folder, file))

    return list(category_map.values()), references

def extract_score(response_text):
    import re
    matches = re.findall(r"\d+(?:\.\d+)?", response_text)
    if matches:
        return float(matches[0])
    return 0.0

def group_video_frames(folder):
    grouped = defaultdict(list)
    for file in os.listdir(folder):
        match = re.match(r"([a-z]+)_\d+\.(jpg|jpeg|png|webp)", file, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            grouped[prefix].append(file)
    return grouped

def classify_group(model, filenames, references, cached_refs, progress):
    total_scores = defaultdict(float)
    total_counts = defaultdict(int)
    full_log = []

    for file in filenames:
        input_path = os.path.join(INPUT_FOLDER, file)
        input_image = lms.prepare_image(input_path)
        full_log.append(f"Classifying frame: {file}")

        for category, ref_paths in references.items():
            score_sum = 0
            num_refs = 0
            for ref_img_path in ref_paths:
                ref_image = cached_refs[ref_img_path]
                prompt = (
                    "You will compare a reference image (one of several from a game) and an input image.\n"
                    "Give a similarity score between 0 (not similar) and 100 (identical).\n"
                    "Only respond with a number."
                )
                chat = lms.Chat()
                chat.add_user_message(prompt, images=[ref_image, input_image])
                response = model.respond(chat)
                value = extract_score(response.content)
                score_sum += value
                num_refs += 1
                progress.update(1)
                full_log.append(f" - [{category}] {os.path.basename(ref_img_path)} → Score: {value:.2f}")

            if num_refs > 0:
                avg_score = score_sum / num_refs
                total_scores[category] += avg_score
                total_counts[category] += 1
                full_log.append(f"   Partial avg for {category}: {avg_score:.2f}\n")

    averaged = {cat: total_scores[cat] / total_counts[cat] for cat in total_scores if total_counts[cat]}
    if not averaged:
        return "others", False, "\n".join(full_log)

    sorted_scores = sorted(averaged.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    is_ambiguous = (best_score - second_score) < threshold
    full_log.append(f"\n→ Final best: {best_category} ({best_score:.2f}), Second: {second_score:.2f}")
    return best_category, is_ambiguous, "\n".join(full_log)

def sort_images_by_reference(model, input_folder, reference_folder, output_folder):
    categories, references = load_categories_and_references(reference_folder)
    all_categories = categories + ["others"]

    cached_refs = {}
    for paths in references.values():
        for ref_img_path in paths:
            if ref_img_path not in cached_refs:
                cached_refs[ref_img_path] = lms.prepare_image(ref_img_path)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs("re", exist_ok=True)
    for category in all_categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)

    frame_groups = group_video_frames(input_folder)
    total_comparisons = sum(len(paths) for paths in references.values()) * sum(len(v) for v in frame_groups.values())
    progress = tqdm(total=total_comparisons, desc="Sorting Progress", unit="img", ncols=80)

    for prefix, files in frame_groups.items():
        category, is_ambiguous, log = classify_group(model, files, references, cached_refs, progress)
        representative = files[0]
        src = os.path.join(input_folder, representative)

        if is_ambiguous:
            dst = os.path.join("re", representative)
            log_path = os.path.join("re", f"{prefix}.txt")
        else:
            dst = os.path.join(output_folder, category, representative)
            log_path = os.path.join(output_folder, category, f"{prefix}.txt")

        shutil.copy(src, dst)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log)
            f.write(f"\nmoved '{representative}' to '{'re' if is_ambiguous else category}'")

    progress.close()

def clear_output_folder(output_folder):
    if not os.path.exists(output_folder):
        print("output folder does not exist")
        return

    for category in os.listdir(output_folder):
        category_path = os.path.join(output_folder, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                os.remove(file_path)
            os.rmdir(category_path)
            print(f"cleared folder: {category_path}")

if __name__ == "__main__":
    model = None
    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == 'clear':
            clear_output_folder(OUTPUT_FOLDER)
            print("output folder cleared")
        else:
            print(f"sorting with references from: {REFERENCE_FOLDER}\nLoading model: {MODEL_ID}")
            start_time = time.time()
            model = lms.llm(MODEL_ID)
            print(f"loaded model in {time.time() - start_time:.2f} seconds")
            second_time = time.time()
            sort_images_by_reference(model, INPUT_FOLDER, REFERENCE_FOLDER, OUTPUT_FOLDER)
            print(f"sorting completed in {time.time() - second_time:.2f} seconds")
    finally:
        if model:
            model.unload()
            print(f"unloaded model: {MODEL_ID}")

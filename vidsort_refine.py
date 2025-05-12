import os
import sys
import shutil
import time
import lmstudio as lms
import re
from collections import defaultdict
from tqdm import tqdm

threshold = 1

REFERENCE_FOLDER = os.path.join(os.getcwd(), 'reference')
INPUT_FOLDER = os.path.join(os.getcwd(), 're')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
MODEL_ID = "gemma-3-27b-it@q6_k"

def load_all_references(ref_folder):
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
    for file in sorted(os.listdir(ref_folder)):
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
    matches = re.findall(r"\d+(?:\.\d+)?", response_text)
    return float(matches[0]) if matches else 0.0

def group_video_frames(folder):
    grouped = defaultdict(list)
    for file in os.listdir(folder):
        match = re.match(r"([a-z]+)_\d+\.(jpg|jpeg|png|webp)", file, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            grouped[prefix].append(file)
    return grouped

def classify_group_adaptive(model, filenames, references_by_category, cached_refs, progress):
    scores_by_cat = defaultdict(float)
    counts_by_cat = defaultdict(int)
    full_log = []

    for file in filenames:
        input_path = os.path.join(INPUT_FOLDER, file)
        input_image = lms.prepare_image(input_path)

        max_refs = max(len(refs) for refs in references_by_category.values())
        num_refs = 1
        while num_refs <= max_refs:
            current_scores = defaultdict(list)
            full_log.append(f"Classifying: {file} using {num_refs} refs\n")

            for category, ref_paths in references_by_category.items():
                for ref_img_path in ref_paths[:num_refs]:
                    ref_image = cached_refs[ref_img_path]

                    prompt = (
                        "You will compare a reference image (from a game) and an input image.\n"
                        "Give a similarity score between 0 (not similar) and 100 (identical).\n"
                        "Only respond with a number."
                    )

                    chat = lms.Chat()
                    chat.add_user_message(prompt, images=[ref_image, input_image])
                    response = model.respond(chat)
                    value = extract_score(response.content)
                    current_scores[category].append(value)
                    full_log.append(f" - [{category}] {os.path.basename(ref_img_path)} → Score: {value}")
                    progress.update(1)

            avg_scores = {cat: sum(vals)/len(vals) for cat, vals in current_scores.items() if vals}
            if not avg_scores:
                break

            sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            best_score = sorted_scores[0][1]
            second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1

            full_log.append(f" → Best: {sorted_scores[0][0]} ({best_score}), Second: {second_score}\n")

            if (best_score - second_score) > threshold or num_refs == max_refs:
                break

            num_refs += 1

        for cat, val in avg_scores.items():
            scores_by_cat[cat] += val
            counts_by_cat[cat] += 1

    final_avg = {cat: scores_by_cat[cat]/counts_by_cat[cat] for cat in scores_by_cat if counts_by_cat[cat]}
    if not final_avg:
        return "others", "\n".join(full_log)

    sorted_final = sorted(final_avg.items(), key=lambda x: x[1], reverse=True)
    best_cat = sorted_final[0][0]
    best_val = sorted_final[0][1]
    second_val = sorted_final[1][1] if len(sorted_final) > 1 else -1

    full_log.append(f"\n→ Final Decision: {best_cat} ({best_val}) | Second: {second_val}")
    return best_cat, "\n".join(full_log)

def sort_images_adaptive(model, input_folder, reference_folder, output_folder):
    categories, references = load_all_references(reference_folder)
    all_categories = categories + ["others"]

    cached_refs = {}
    for ref_paths in references.values():
        for ref_img_path in ref_paths:
            if ref_img_path not in cached_refs:
                cached_refs[ref_img_path] = lms.prepare_image(ref_img_path)

    os.makedirs(output_folder, exist_ok=True)
    for category in all_categories:
        os.makedirs(os.path.join(output_folder, category), exist_ok=True)

    grouped = group_video_frames(input_folder)
    progress = tqdm(total=sum(len(v)*len(references) for v in grouped.values()), desc="Adaptive Sort", unit="img", ncols=80)

    for prefix, frames in grouped.items():
        category, log = classify_group_adaptive(model, frames, references, cached_refs, progress)

        os.makedirs(os.path.join(output_folder, category), exist_ok=True)
        rep_frame = frames[0]
        dest_img = os.path.join(output_folder, category, rep_frame)
        dest_txt = os.path.join(output_folder, category, f"{prefix}.txt")

        shutil.move(os.path.join(input_folder, rep_frame), dest_img)
        for other in frames:
            if other != rep_frame:
                os.remove(os.path.join(input_folder, other))
            txt = os.path.splitext(other)[0] + '.txt'
            txt_path = os.path.join(input_folder, txt)
            if os.path.exists(txt_path):
                os.remove(txt_path)

        with open(dest_txt, 'w', encoding='utf-8') as log_file:
            log_file.write(log)
            log_file.write(f"\nRefined move: '{rep_frame}' → '{category}/'\n")

    progress.close()

if __name__ == "__main__":
    print(f"redoing sort from: {INPUT_FOLDER}\nusing references in: {REFERENCE_FOLDER}")
    start_time = time.time()
    model = lms.llm(MODEL_ID)
    print(f"loaded model in {time.time() - start_time:.2f} seconds")
    second_time = time.time()
    sort_images_adaptive(model, INPUT_FOLDER, REFERENCE_FOLDER, OUTPUT_FOLDER)
    print(f"processed in {time.time() - second_time:.2f} seconds")
    model.unload()
    print(f"unloaded model: {MODEL_ID}")

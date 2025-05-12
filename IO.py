import os
import cv2
import shutil
import string
import json
import itertools
import csv

VIDEO_INPUT_DIR = "vinput"
FRAME_INPUT_DIR = "input"
FRAME_OUTPUT_DIR = "output"
VIDEO_OUTPUT_DIR = "voutput"
FRAME_SUFFIX = "_frame.jpg"
MAPPING_FILE = os.path.join(VIDEO_INPUT_DIR, "frame_video_map.json")


def extract_three_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    target_indices = [0, total_frames // 2, total_frames - 1]
    frames = []

    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue

        h, w, _ = frame.shape
        crop_size = 896
        bottom = h
        top = max(h - crop_size, 0)
        left = max((w - crop_size) // 2, 0)
        right = left + crop_size

        cropped = frame[top:bottom, left:right]
        frames.append(cropped)

    cap.release()
    return frames



def clear_folder_empty(folder):
    return not any(os.scandir(folder))

def get_all_videos_with_rel_path(base_dir):
    all_videos = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                all_videos.append(rel_path)
    return sorted(all_videos)

def generate_alpha_names(n):
    result = []
    i = 1
    while len(result) < n:
        for combo in itertools.product(string.ascii_lowercase, repeat=i):
            result.append(''.join(combo))
            if len(result) == n:
                return result
        i += 1
    return result

def mode_1_generate_frames():
    print("extracting frames from videos")
    videos = get_all_videos_with_rel_path(VIDEO_INPUT_DIR)
    labels = generate_alpha_names(len(videos))

    name_map = {}
    for idx, rel_video_path in enumerate(videos):
        full_video_path = os.path.join(VIDEO_INPUT_DIR, rel_video_path)
        frames = extract_three_frames(full_video_path)
        label = labels[idx]

        for i, frame in enumerate(frames):
            if frame is None:
                continue
            frame_name = f"{label}_{i}.jpg"
            cv2.imwrite(os.path.join(FRAME_INPUT_DIR, frame_name), frame)
            name_map[frame_name] = rel_video_path.replace("\\", "/")
        if all(f is None for f in frames):
            print(f"skipping video: {rel_video_path}")

    with open(MAPPING_FILE, "w") as f:
        json.dump(name_map, f, indent=2)

    print("frames saved")

def mode_2_sort_videos():
    print("reorganizing")

    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError("no mapping found")

    with open(MAPPING_FILE, "r") as f:
        frame_to_video = json.load(f)

    csv_rows = []

    for category in os.listdir(FRAME_OUTPUT_DIR):
        category_path = os.path.join(FRAME_OUTPUT_DIR, category)
        if not os.path.isdir(category_path):
            continue

        dest_dir = os.path.join(VIDEO_OUTPUT_DIR, category)
        os.makedirs(dest_dir, exist_ok=True)

        for frame_file in os.listdir(category_path):
            if frame_file in frame_to_video:
                rel_video_path = frame_to_video[frame_file]
                src = os.path.join(VIDEO_INPUT_DIR, rel_video_path)
                dst = os.path.join(dest_dir, os.path.basename(rel_video_path))
                shutil.copy2(src, dst)
                print(f"[COPIED] {rel_video_path} -> {dest_dir}")
                csv_rows.append([frame_file, rel_video_path, category])
            else:
                print(f"no matching video for frame: {frame_file}")

    csv_file = "decision_summary.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame File", "Video Path", "Assigned Category"])
        writer.writerows(csv_rows)

    print(f"videos sorted, decision log written")

os.makedirs(FRAME_INPUT_DIR, exist_ok=True)
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)

if clear_folder_empty(FRAME_INPUT_DIR) and clear_folder_empty(FRAME_OUTPUT_DIR):
    mode_1_generate_frames()
else:
    mode_2_sort_videos()

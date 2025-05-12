import os
import random
import string
import uuid

folder_path = r"C:\Users\brian\Desktop\LMSorter\input"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

random.shuffle(files)

temp_map = {}
for filename in files:
    old_path = os.path.join(folder_path, filename)
    temp_name = f"temp_{uuid.uuid4().hex}{os.path.splitext(filename)[1]}"
    temp_path = os.path.join(folder_path, temp_name)
    os.rename(old_path, temp_path)
    temp_map[temp_name] = os.path.splitext(filename)[1]

def generate_alphabet_names(n):
    result = []
    i = 1
    while len(result) < n:
        for combo in map(''.join, __import__('itertools').product(string.ascii_lowercase, repeat=i)):
            result.append(combo)
            if len(result) == n:
                break
        i += 1
    return result

alphabet_names = generate_alphabet_names(len(temp_map))

for temp_name, alpha in zip(temp_map.keys(), alphabet_names):
    ext = temp_map[temp_name]
    final_name = f"{alpha}{ext}"
    os.rename(
        os.path.join(folder_path, temp_name),
        os.path.join(folder_path, final_name)
    )

print("files renamed")

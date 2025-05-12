import subprocess
import time
import os
import sys
import shutil

def clear_folders():
    folders = ["input", "output", "re", "voutput"]
    for folder in folders:
        if os.path.exists(folder):
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"cleared {folder}")

def run_script(script_name):
    print(f"\nstarting {script_name}")
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_folders()
    else:
        scripts = ["IO.py", "vidsort.py", "vidsort_refine.py", "IO.py"]
        start = time.time()
        for script in scripts:
            run_script(script)
        print(f"total time finished in {time.time() - start:.2f} seconds")

import os
import re

# The folders you actually touched. DO NOT add 'thirdparty' or 'core' if you didn't touch them!
TARGET_DIRS = ['drivers/gles3', 'scene/3d', 'scene/main', 'servers/visual']

# The AI junk patterns you want to annihilate
JUNK_PATTERNS = [
    re.compile(r'//\s*<---.*ADD THIS'),
    re.compile(r'//\s*<---.*WE ADDED THIS'),
    re.compile(r'//\s*<---.*MOVED HERE!'),
    re.compile(r'//\s*-{10,}'),          # Matches // -------------
]

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        if any(pattern.search(line) for pattern in JUNK_PATTERNS):
            modified = True
            continue # Skip this line
        new_lines.append(line)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Cleaned: {filepath}")

for d in TARGET_DIRS:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith(('.cpp', '.h', '.glsl', '.xml')):
                clean_file(os.path.join(root, file))

print("Sweep complete.")

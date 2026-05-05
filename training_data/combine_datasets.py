"""
Combine all training datasets into one master file.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "teranga_finetune_combined.jsonl")

files = [
    os.path.join(BASE_DIR, "teranga_finetune.jsonl"),
    os.path.join(BASE_DIR, "teranga_finetune_advanced.jsonl"),
]

total = 0
with open(OUTPUT, "w", encoding="utf-8") as out:
    for f in files:
        with open(f, "r", encoding="utf-8") as inp:
            for line in inp:
                if line.strip():
                    out.write(line)
                    total += 1

print(f"Combined {total} examples into {OUTPUT}")

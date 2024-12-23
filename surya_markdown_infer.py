import os

from config import INPUT_DIR, OUTPUT_DIR

files = os.listdir(INPUT_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for file in files:
    command = f"marker_single pdfs/{file} --output_dir {OUTPUT_DIR} --output_format markdown --force_ocr"
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR

model = ocr_predictor(pretrained=True)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)




for file in tqdm(os.listdir(INPUT_DIR)):
    # PDF
    doc = DocumentFile.from_pdf(INPUT_DIR + file)
    # Analyze
    result = model(doc)

    json_output = result.export()

    string_result = result.render()
    print(string_result)

    
    name = file.split('.')[0]
    with open(OUTPUT_DIR + name + '.txt', 'w') as f:
        f.write(string_result)
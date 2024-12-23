from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
from tqdm import tqdm

model = ocr_predictor(pretrained=True)


data_dir = "pdfs"
output_dir = "doc_outs"


for file in tqdm(os.listdir(data_dir)):
    # PDF
    doc = DocumentFile.from_pdf(data_dir + file)
    # Analyze
    result = model(doc)

    json_output = result.export()

    string_result = result.render()
    print(string_result)

    
    name = file.split('.')[0]
    with open(output_dir + name + '.txt', 'w') as f:
        f.write(string_result)
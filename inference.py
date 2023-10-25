import os
import pandas as pd
import torch
import fastwer
from tqdm import tqdm

from evaluate import load
from PIL import Image
from transformers import (VisionEncoderDecoderModel, ViTImageProcessor, ByT5Tokenizer,TrOCRProcessor, ViTModel)


torch.cuda.empty_cache()
device = torch.device('cuda')

from custom_class import T5DecoderOnlyForCausalLM
from config import ENCODER, DECODER, LANGUAGE, MAX_TOKENS

def calculate_crr_wrr(results, ground_truth):
    cer = fastwer.score(results, ground_truth, char_level=True)
    wer = fastwer.score(results, ground_truth)
    return 100- cer, 100-wer

def preview(image_path, model, processor, device, text):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    model = model.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=MAX_TOKENS, num_beams = 4)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text



MODEL_PATH = '/home/ganesh/BADRI/TRANSFORMERS/checkpoints/hindi/checkpoint-2000/'
RESULTS_DIR = './../results/'
DATA_PATH =  '/home/ganesh/BADRI/RECOGNITION/data/iiit_indic_words/hindi/'

cer_metric = load('cer')

tokenizer = ByT5Tokenizer.from_pretrained(ENCODER)
image_processor=ViTImageProcessor.from_pretrained(DECODER)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


encoder = ViTModel.from_pretrained(ENCODER)
decoder = T5DecoderOnlyForCausalLM.from_pretrained(DECODER)
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'pytorch_model.bin')))
model.eval()

preds, gts = [], []
        
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.txt'), names=['file_name', 'text'], sep=' ')
for _, row in tqdm(test_df.iterrows()):
    image_path = DATA_PATH + 'test/' + row['file_name']
    generated_text = preview(image_path, model, processor, device, row['text'])
    preds.append(generated_text)
    gts.append(row['text'])
    
data = {'preds': preds, 'actual': gts}
df = pd.DataFrame(data)
print(calculate_crr_wrr(preds, gts))
df.to_csv(RESULTS_DIR + LANGUAGE + '.csv', index=False, header=False)
    


    

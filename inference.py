import os
import pandas as pd
import torch
import fastwer
from tqdm import tqdm
import csv

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


def save_to_csv(data, filename='./../results/bengali_data.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


MODEL_PATH = '/home/ganesh/BADRI/RECOGNITION/BYT5/checkpoints/merged/checkpoint-'
RESULTS_DIR = './../results/'
DATA_PATH =  '/home/ganesh/BADRI/merged/'


cer_metric = load('cer')


tokenizer = ByT5Tokenizer.from_pretrained(DECODER)
image_processor=ViTImageProcessor.from_pretrained(ENCODER)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


encoder = ViTModel.from_pretrained(ENCODER)
decoder = T5DecoderOnlyForCausalLM.from_pretrained(DECODER)
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

checkpoints = [12,6]

for checkpoint in checkpoints:

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH + str(checkpoint*10000) + '/', 'pytorch_model.bin')))

    model.to(device)
    model.eval()


    langs = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

    for lang_curr in langs:
        preds, gts = [], []

        FILE_PATH = f'/home/ganesh/BADRI/merged/langs/{lang_curr}.txt'
        test_df = pd.read_csv(os.path.join(FILE_PATH), names=['file_name', 'text'], sep=' ')
        if not os.path.exists(f'./../results/{checkpoint}/'):
            os.makedirs(f'./../results/{checkpoint}/')
        for _, row in tqdm(test_df.iterrows()):
            image_path = DATA_PATH + 'test/' + row['file_name']
            generated_text = preview(image_path, model, processor, device, row['text'])
            preds.append(generated_text)
            gts.append(row['text'])
            save_to_csv([generated_text, row['text']], filename=f'./../results/{checkpoint}/{lang_curr}.csv')
        


    

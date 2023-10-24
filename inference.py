import os
import pandas as pd
import torch

torch.cuda.set_device(0)
torch.cuda.empty_cache()
device = torch.device('cuda')


from datasets import load_metric
from PIL import Image

from transformers import (VisionEncoderDecoderModel, ViTImageProcessor, ByT5Tokenizer,Seq2SeqTrainer, Seq2SeqTrainingArguments,TrOCRProcessor, ViTModel, default_data_collator)


import warnings
warnings.filterwarnings("ignore")

from custom_class import T5DecoderOnlyForCausalLM
from config import *

cer_metric = load_metric('cer')

tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
image_processor=ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


encoder = ViTModel.from_pretrained(ENCODER)
decoder = T5DecoderOnlyForCausalLM.from_pretrained(DECODER)
# Create the VisionEncoderDecoderModel with the loaded encoder and decoder
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)


model_path = '/home/ganesh/BADRI/TRANSFORMERS/checkpoints/hindi/checkpoint-2000/'
model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))

# torch.save(model.state_dict(), os.path.join('./models/', 'model_state_dict.pth'))

model.eval()


def preview(image_path, model, processor, device, text):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    model = model.to(device)
    generated_ids = model.generate(pixel_values=pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Generated text:", generated_text,"\tActual text:", text)
    return generated_text
    
    
dataset_path = '/home/ganesh/BADRI/RECOGNITION/data/iiit_indic_words/hindi/'
    
   
preds, gts = [], []
        
test_df = pd.read_csv(os.path.join(dataset_path, 'test.txt'), names=['file_name', 'text'], sep=' ')
for _, row in test_df.iterrows():
    image_path = dataset_path + 'test/' + row['file_name']
    generated_text = preview(image_path, model, processor, device, row['text'])
    preds.append(generated_text)
    gts.append(row['text'])
    
data = {'preds': preds, 'actual': gts}
df = pd.DataFrame(data)

df.to_csv('./../results/' + LANGUAGE + '.csv', index=False, header=False)
    


    

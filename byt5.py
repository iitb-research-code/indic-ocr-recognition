import os
import pandas as pd
import torch

torch.cuda.set_device(1)
torch.cuda.empty_cache()
device = torch.device('cuda:1')


from datasets import load_metric
from PIL import Image

from transformers import (VisionEncoderDecoderModel, ViTImageProcessor, ByT5Tokenizer,Seq2SeqTrainer, Seq2SeqTrainingArguments,TrOCRProcessor, ViTModel, default_data_collator)


import warnings
warnings.filterwarnings("ignore")

from t5 import T5DecoderOnlyForCausalLM

cer_metric = load_metric('cer')



output_folder = './results/'
dataset_path = '/data/BADRI/DATASETS/BENCHMARK/RECOGNITON/iiit_indic_words/hindi/'

exp_type = "crs_loss"
batch_size = 1



class IAMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # print(encoding)
        return encoding


    
train_df = pd.read_csv(os.path.join(dataset_path, 'train.txt'), names=['file_name', 'text'], sep=' ', nrows=8000)
val_df = pd.read_csv(os.path.join(dataset_path, 'val.txt'), names=['file_name', 'text'], sep=' ', nrows=1000)


tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-base')
image_processor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


# Initialize the dataset and dataloader
train_dataset = IAMDataset(root_dir=dataset_path, df=train_df, processor=processor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = IAMDataset(root_dir=dataset_path, df=val_df, processor=processor)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
decoder = T5DecoderOnlyForCausalLM.from_pretrained("google/byt5-base")

model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4



training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    num_train_epochs=500,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=False, ##
    fp16_full_eval=False,  # Disable FP16 full evaluation
    output_dir="./checkpoints/",
    # logging_steps=500,
    # save_steps=500,
    # eval_steps=500,
)


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}



trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)


trainer.train()
model.save_pretrained('./models/')
torch.save(model.state_dict(), os.path.join('./models/', 'model_state_dict.pth'))


# encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
# decoder = T5DecoderOnlyForCausalLM.from_pretrained("google/byt5-base")
# # Create the VisionEncoderDecoderModel with the loaded encoder and decoder
# model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
# model_path = '/content/drive/MyDrive/IITB/testbyt5'
# model.load_state_dict(torch.load(os.path.join(model_path, 'model_state_dict.pth')))
# model.eval()


def preview(image_path, model, processor, device, text):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    model = model.to(device)
    generated_ids = model.generate(pixel_values=pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Generated text:", generated_text, text)
    
    
    
        
    
for _, row in val_df.iterrows():
    image_path = dataset_path + row['file_name']
    preview(image_path, model, processor, device, row['text'])

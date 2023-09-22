import os
import torch
torch.cuda.empty_cache()

from PIL import Image
import pandas as pd

from torch.utils.data import Dataset
from transformers import (ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator)
from evaluate import load

from config import *

import warnings
warnings.filterwarnings('ignore')


class OCRDataset(Dataset):
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
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # print(encoding)
        return encoding
    
    
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def dataset_generator(root_dir):
    train_df = pd.read_csv(root_dir + 'train.txt', sep=' ', names = ['file_name', 'text'])
    val_df   = pd.read_csv(root_dir + 'val.txt', sep=' ', names = ['file_name', 'text'])
    return train_df, val_df
    
    
if __name__ == "__main__":
    
    train_df, val_df = dataset_generator(DATA_DIR)
    print(f"Train & Val shape: {train_df.shape, val_df.shape}")

    feature_extractor=ViTFeatureExtractor.from_pretrained(ENCODER)
    tokenizer = RobertaTokenizer.from_pretrained(DECODER)
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    train_dataset = OCRDataset(root_dir=DATA_DIR+'train/', df=train_df, processor=processor)
    eval_dataset  = OCRDataset(root_dir=DATA_DIR+'val/', df=val_df, processor=processor)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER, DECODER)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    print(f"processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}")
    model.config.vocab_size = model.config.decoder.vocab_size
    # config_decoder.is_decoder = True
    # config_decoder.add_cross_attention = True

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        evaluation_strategy="steps",
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=2,
        save_steps=2000,
        eval_steps=100,
    )

    cer_metric = load("cer")



    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    os.makedirs("model/")
    model.save_pretrained("model/")
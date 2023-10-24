import os
import json
import pandas as pd
import torch

from evaluate import load
from transformers import (VisionEncoderDecoderModel, ViTImageProcessor, ByT5Tokenizer,Seq2SeqTrainer, Seq2SeqTrainingArguments,TrOCRProcessor, ViTModel, default_data_collator)


from custom_class import T5DecoderOnlyForCausalLM, OCRDataset
from config import *

torch.cuda.empty_cache()
device = torch.device('cuda')

cer_metric = load('cer')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def dataset_generator(root_dir):   
    
    with open(root_dir + 'train.json', 'r') as json_file:
        data = json.load(json_file)
    
    train_df = pd.DataFrame.from_dict(data, orient ='index').reset_index()
    train_df.columns = ['file_name', 'text']
    
    with open(root_dir + 'val.json', 'r') as json_file:
        data = json.load(json_file)
    
    val_df = pd.DataFrame.from_dict(data, orient ='index').reset_index()
    val_df.columns = ['file_name', 'text']
    
    return train_df, val_df


if __name__ == "__main__":
    
    train_df, val_df = dataset_generator(DATA_PATH)
    print(f"Train & Val shape: {train_df.shape, val_df.shape}")


    tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
    image_processor=ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


    # Initialize the dataset and dataloader
    train_dataset = OCRDataset(root_dir=DATA_PATH + 'train/', df=train_df, processor=processor)
    val_dataset = OCRDataset(root_dir=DATA_PATH + 'val/', df=val_df, processor=processor)


    encoder = ViTModel.from_pretrained(ENCODER)
    decoder = T5DecoderOnlyForCausalLM.from_pretrained(DECODER)

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
    model.config.decoder.max_length = 64
    model.config.encoder.max_length = 64


    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=False, ##
        fp16_full_eval=False,  # Disable FP16 full evaluation
        output_dir=CHECKPOINTS_DIR,
        logging_steps=2000,
        save_steps=2000,
        eval_steps=2000
    )
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        

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
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model_state_dict.pth'))

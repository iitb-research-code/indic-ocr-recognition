from transformers import (ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel)
from PIL import Image
from config import ENCODER, DECODER, CHECKPOINT_FILE, DATA_DIR


feature_extractor=ViTFeatureExtractor.from_pretrained(ENCODER)
tokenizer = RobertaTokenizer.from_pretrained(DECODER)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_FILE)

def preview(image_path, image_name, original_text):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text, ":\t", image_name, ":\t", original_text)


test_df  = pd.read_csv(DATA_DIR + 'test.txt', sep=' ', names = ['file_name', 'text'])
for _, row in test_df.iterrows():
    preview(DATA_DIR + 'test/' + row['file_name'], row['file_name'], row['text'])
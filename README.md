# indic-ocr-recognition
OCR models Development and hosting all the available trained models of IIT Bombay


# Installation

```pip install -r requirements.txt```


# Training

1. Update all the hyperparameters and data inputs through *config.py* file
2. ```python train.py``` for training


# Architectures

## Encoder 

1. google/vit-base-patch16-224
2. google/vit-base-patch32-384

## Decoder 

1. google/byt5-small
2. google/byt5-base


# Results

CRR - 94.23
WRR - 69.78
 
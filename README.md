# Indic TrOCR - Transformer based OCR for Indian Languages

TrOCR is an OCR (Optical Character Recognition) model proposed by Minghao Li et al. in their paper titled <a href="https://arxiv.org/abs/2109.10282">TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a>. 
This model is composed of an image Transformer encoder and an autoregressive text Transformer decoder, enabling it to accurately perform OCR.

In this repository, you will find TrOCR, an OCR model specifically developed for recognizing handwritten Indian documents in various Indian languages for handwritten text specially. The TrOCR model has been designed to accurately detect and convert text in these languages from images of handwritten documents, making it a valuable tool for various applications such as digitizing old documents, extracting information from scanned documents, and more.

## Installation

```
virtualenv trocr_env
source trocr_env/bin/activate
git clone https://github.com/iitb-research-code/indic-ocr-recognition.git
cd indic-ocr-recognition
git checkout trocr
pip install -r requirements.txt
```


## Dataset Details

1. The Training on TROCR model has been performed on [IIIT-HW Dataset](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)
2. The format of the data is as follows
```
DATA
├── hindi
|   └──train/
|   └──test/
|   └──val/
|   └──train.txt
|   └──test.txt
|   └──val.txt
|   └──vocab.txt
```

## Train for a new Language

1. Make all configuration changes from *config.py* file as necessary
2. To train the TROCR model, run the following command
```python train.py```
```

## Inference Steps

1. To evaluate and get inference results, run the following command after configurations in *config.py* file accordingly
```
python inference.py
```
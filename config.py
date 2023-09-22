DATA_DIR = "/raid/nlp/ganesh/ocr/DATA/hindi/"

ENCODER = 'google/vit-base-patch16-224-in21k'
DECODER = 'flax-community/roberta-hindi'

EPOCHS = 10
BATCH_SIZE = 16
DEVICE = 5

CHECKPOINT_PATH = './checkpoints/'
CHECKPOINT_FILE = CHECKPOINT_PATH + 'checkpoint-5400'
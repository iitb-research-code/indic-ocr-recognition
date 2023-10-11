DATA_DIR = "/home/venkat/BADRI/RECOGNITION/data/iiit_indic_words/gujarati/"

ENCODER = 'google/vit-base-patch16-224-in21k'
DECODER = 'flax-community/roberta-hindi'

EPOCHS = 100
BATCH_SIZE = 16
DEVICE = 4

RESUME = False

MODEL_FILE = ''

CHECKPOINT_PATH = './checkpoints/'
CHECKPOINT_FILE = CHECKPOINT_PATH + 'checkpoint-5400'
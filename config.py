DATA_DIR = '/data/BADRI/RECOGNITION/datasets/'
LANGUAGE = 'hindi_new'

DATA_PATH = DATA_DIR + LANGUAGE + '/'
MODEL_DIR = './../models/' + LANGUAGE + '/'
CHECKPOINTS_DIR = './../checkpoints/' + LANGUAGE + '/'


BATCH_SIZE = 64
EPOCHS = 50
MAX_TOKENS = 40


ENCODER = "google/vit-base-patch32-384"
DECODER = "google/byt5-small"

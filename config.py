DATA_DIR = '/data/BADRI/DATASETS/BENCHMARK/RECOGNITION/handwritten/'
LANGUAGE = 'merged'

DATA_PATH = DATA_DIR + LANGUAGE + '/'
MODEL_DIR = './../models/' + LANGUAGE + '/'
CHECKPOINTS_DIR = './../checkpoints/' + LANGUAGE + '/'


BATCH_SIZE = 16
EPOCHS = 50
MAX_TOKENS = 40


ENCODER = "google/vit-base-patch32-384"
DECODER = "google/byt5-small"

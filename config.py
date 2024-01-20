DATA_DIR = '/home/ganesh/BADRI/synth/generated/'
LANGUAGE = 'hindi'

DATA_PATH = DATA_DIR + LANGUAGE + '/'
MODEL_DIR = './../models/' + LANGUAGE + '/'
CHECKPOINTS_DIR = './../checkpoints/' + LANGUAGE + '/'


BATCH_SIZE = 256
EPOCHS = 30
MAX_TOKENS = 32

DECODER_BLOCKS = 1

SAVE_STEPS = 1000
EVAL_STEPS = 5000


ENCODER = "google/vit-base-patch16-224"
# ENCODER = "google/vit-base-patch32-384"
DECODER = "google/byt5-small"

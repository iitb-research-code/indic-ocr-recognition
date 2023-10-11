DATA_DIR = '/home/venkat/BADRI/RECOGNITION/data/iiit_indic_words/'
LANGUAGE = 'bengali'

DATA_PATH = DATA_DIR + LANGUAGE + '/'
MODEL_DIR = './../models/' + LANGUAGE + '/'
CHECKPOINTS_DIR = './../checkpoints/' + LANGUAGE + '/'



BATCH_SIZE = 16
EPOCHS = 100
DEVICE = 5


ENCODER = "google/vit-base-patch16-224"
DECODER = "google/byt5-base"
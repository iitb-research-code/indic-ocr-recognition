DATA_DIR = '/home/ganesh/BADRI/RECOGNITION/data/iiit_indic_words/'
LANGUAGE = 'hindi'

DATA_PATH = DATA_DIR + LANGUAGE + '/'
MODEL_DIR = './../models/' + LANGUAGE + '_test/'
CHECKPOINTS_DIR = './../checkpoints/' + LANGUAGE + '_test/'



BATCH_SIZE = 64
EPOCHS = 50
DEVICE = 5


ENCODER = "google/vit-base-patch32-384"
DECODER = "google/byt5-small"

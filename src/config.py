import os

# DAVIS Paths
ANNOTATIONS_FOLDERS_PATH = "DAVIS_2016/DAVIS/Annotations/480p/"
ANNOTATIONS_AUGMENTED_FOLDERS_PATH = "DAVIS_2016/DAVIS/Annotations_augmented/480p/"

IMAGES_FOLDERS_PATH = 'DAVIS_2016/DAVIS/JPEGImages/480p/'
IMAGES_AUGMENTED_FOLDERS_PATH = 'DAVIS_2016/DAVIS/JPEGImages_augmented/480p/'

# PG Dataset Paths
PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH = 'pg_datasets/DAVIS_2016'

CONTOURS_FOLDERS_PATH = os.path.join(PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                                    'Contours')
TRANSLATIONS_FOLDERS_PATH = os.path.join(PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                                    'Translations')
# Results Paths
OSVOS_RESULTS_FOLDERS_PATH = 'OSVOS_PyTorch/models/Results_OSVOS_Tim/vgg_test_seqs/'
COMBO_RESULTS_FOLDERS_PATH = 'evaluations/'

# Sequences
SKIP_SEQUENCES = []

TRAIN_SEQUENCES = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 
                   'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 
                   'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 
                   'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 
                   'motocross-bumps', 'motorbike', 'paragliding', 'rhino', 
                   'rollerblade', 'scooter-gray', 'soccerball', 'stroller',
                   'surf', 'swing', 'tennis', 'train']

VAL_SEQUENCES = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',
                 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 
                 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 
                 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 
                 'soapbox']

DEBUG = 2

# Data Augmentation
AUGMENTATION_COUNT = 2
MEANVAL = (104.00699, 116.66877, 122.67892)

# Contour and Translation Creation
CLOSING_KERNEL_SIZE = 25

# Dataset Creation
EPOCHS_WO_AVEGRAD = 200
LAYER = 9
K = 32
NUM_SEQUENCES = 5

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 20

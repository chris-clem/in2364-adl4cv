"""Third step of our approach: create Pytorch Geometric train and val dataset.

Takes a long time to run if you create a new one.
"""

import src.config as cfg
from src.davis_2016 import DAVIS2016


if __name__ == "__main__": 
    train = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                      cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, cfg.CONTOURS_FOLDERS_PATH, 
                      cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                      cfg.PARENT_MODEL_PATH,
                      cfg.LAYER, cfg.K, cfg.AUGMENTATION_COUNT,
                      cfg.SKIP_SEQUENCES, 
                      cfg.TRAIN_SEQUENCES, cfg.VAL_SEQUENCES,
                      train=True)


    val = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                    cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, cfg.CONTOURS_FOLDERS_PATH, 
                    cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                    cfg.PARENT_MODEL_PATH,
                    cfg.LAYER, cfg.K, 0,
                    cfg.SKIP_SEQUENCES, 
                    cfg.TRAIN_SEQUENCES, cfg.VAL_SEQUENCES,
                    train=False)


    print("Train size: %i" % len(train))
    print("Val size: %i" % len(val))
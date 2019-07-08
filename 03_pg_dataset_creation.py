import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler
from torch_geometric.data import DataLoader

from pg_networks.dynamic_edge import DynamicEdge
from pg_networks.gcn import GCN
from pg_networks.sg import SG
import src.config as cfg
from src.davis_2016 import DAVIS2016
from src.solver import Solver
from src.vis_utils import plot_img_with_contour_and_translation, plot_translations, plot_loss, \
                          plot_combo_img

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                  cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, cfg.CONTOURS_FOLDERS_PATH, 
                  cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                  cfg.PARENT_MODEL_PATH,
                  cfg.LAYER, cfg.K, cfg.EPOCHS_WO_AVEGRAD, cfg.AUGMENTATION_COUNT,
                  cfg.SKIP_SEQUENCES, cfg.TRAIN_SEQUENCES, cfg.VAL_SEQUENCES,
                  train=True)
                  
val = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                  cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, cfg.CONTOURS_FOLDERS_PATH, 
                  cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                  cfg.PARENT_MODEL_PATH,
                  cfg.LAYER, cfg.K, cfg.EPOCHS_WO_AVEGRAD, 0,
                  cfg.SKIP_SEQUENCES, cfg.TRAIN_SEQUENCES, cfg.VAL_SEQUENCES,
                  train=False)
                  
print("Train size: %i" % len(train))
print("Val size: %i" % len(val))
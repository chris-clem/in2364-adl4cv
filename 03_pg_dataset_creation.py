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

train_loader = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=cfg.BATCH_SIZE, shuffle=False)

# Load model and run the solver
model = GCN(in_channels=train[0].num_features, 
            out_channels=train[0].y.shape[1])

solver = Solver(optim_args={"lr": cfg.LEARNING_RATE,
                            "weight_decay": cfg.WEIGHT_DECAY})

solver.train(model, train_loader, val_loader,
             num_epochs=cfg.NUM_EPOCHS, log_nth=100, verbose=True)

torch.save(model.state_dict(), 'pg_models/trained_model.pth')

#Val size: 1356
# START TRAIN.
# Traceback (most recent call last):
#   File "03_pg_dataset_creation.py", line 56, in <module>
#     num_epochs=cfg.NUM_EPOCHS, log_nth=100, verbose=True)
#   File "/home/maximilian_boemer/in2364-adl4cv/src/solver.py", line 102, in trai
# n
#     for i, data in enumerate(train_loader):
#   File "/opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.
# py", line 560, in __next__
#     batch = self.collate_fn([self.dataset[i] for i in indices])
#   File "/opt/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.
# py", line 560, in <listcomp>
#     batch = self.collate_fn([self.dataset[i] for i in indices])
#   File "/home/maximilian_boemer/.local/lib/python3.7/site-packages/torch_geomet
# ric/data/dataset.py", line 124, in __getitem__
#     data = self.get(idx)
#   File "/home/maximilian_boemer/in2364-adl4cv/src/davis_2016.py", line 213, in 
# get
#     data = torch.load(data_path)
#   File "/opt/anaconda3/lib/python3.7/site-packages/torch/serialization.py", lin
# e 382, in load
#     f = open(f, 'rb')
# FileNotFoundError: [Errno 2] No such file or directory: 'pg_datasets/DAVIS_2016
# /processed/paragliding_100_00054.pt'
# maximilian_boemer@adl4cv-vm:~/in2364-adl4cv$
"""Forth step of our approach: train a GCN."""

from torch_geometric.data import DataLoader

from pg_networks.gcn import GCN
from pg_networks.dynamic_edge import DynamicEdge
import src.config as cfg
from src.davis_2016 import DAVIS2016
from src.solver import Solver
from src.vis_utils import save_loss


if __name__ == "__main__": 
    # Train and val dataset
    train = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                      cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, 
                      cfg.CONTOURS_FOLDERS_PATH, 
                      cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                      cfg.PARENT_MODEL_PATH,
                      cfg.LAYER, cfg.K, cfg.AUGMENTATION_COUNT,
                      cfg.SKIP_SEQUENCES, 
                      cfg.TRAIN_SEQUENCES[:cfg.NUM_TRAIN_SEQUENCES], 
                      cfg.VAL_SEQUENCES[:cfg.NUM_VAL_SEQUENCES],
                      train=True)

    val = DAVIS2016(cfg.PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH,
                    cfg.ANNOTATIONS_AUGMENTED_FOLDERS_PATH, 
                    cfg.CONTOURS_FOLDERS_PATH, 
                    cfg.IMAGES_AUGMENTED_FOLDERS_PATH, cfg.TRANSLATIONS_FOLDERS_PATH,
                    cfg.PARENT_MODEL_PATH,
                    cfg.LAYER, cfg.K, 0,
                    cfg.SKIP_SEQUENCES, 
                    cfg.TRAIN_SEQUENCES[:cfg.NUM_TRAIN_SEQUENCES], 
                    cfg.VAL_SEQUENCES[:cfg.NUM_VAL_SEQUENCES],
                    train=False)
    
    print("Train size: %i" % len(train))
    print("Val size: %i" % len(val))

    # Train and val Dataloader
    train_loader = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Load model and run the solver
    model = GCN(in_channels=train[0].num_features, 
                out_channels=train[0].y.shape[1])

    solver = Solver(optim_args={"lr": cfg.LEARNING_RATE,
                                "weight_decay": cfg.WEIGHT_DECAY})

    solver.train(model, train_loader, val_loader,
                 num_epochs=cfg.NUM_EPOCHS, log_nth=1000 , verbose=True)
    
    save_loss(solver)


"""PyTorch Geometric custom dataset class for DAVIS 2016 data.

The implementation is based on this tutorial: 
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
"""

import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

import src.config as cfg
from src.create_data import create_osvos_model, create_data
import OSVOS_PyTorch.networks.vgg_osvos as vo


class DAVIS2016(Dataset):
    """PyTorch Geometric custom dataset class for DAVIS 2016 data."""
    
    def __init__(self, root, 
                 annotations_folders_path, contours_folders_path, 
                 images_folders_path, translations_folders_path,
                 parent_model_path,
                 layer, k, augmentation_count,
                 skip_sequences, train_sequences, val_sequences,
                 train=True):
        """Constrcutor for PyTorch Geometric custom dataset class for DAVIS 2016 data.

        Parameters
        ----------
        root : str
            Path to root folder where dataset is stored
        annotations_folders_path : str
            Path to folder where augmented DAVIS 2016 annotations are stored
        contours_folders_path : str
            Path to folder where computed contours are stored
        images_folders_path : str
            Path to folder where augmented DAVIS 2016 images are stored
        translations_folders_path : str
            Path to folder where computed translations are stored
        parent_model_path : str
            Path to trained OSVOS parent model which is used to extract OSVOS feature vectors
        layer : int
            Layer from which to extract OSVOS feature vector (1, 4, 9, 16 are useful values)
        k : int
            Number of neighbours to compute KNN graph of contour points
        augmentation_count : int
            Number of augmentations
        skip_sequences : list
            List of sequences to skip
        train_sequences : list
            List of DAVIS 2016 train sequences
        val_sequences : list
            List of DAVIS 2016 val sequences
        train : bool
            Flag to indicate whether dataset is for train or val
        """

        # Paths
        self.annotations_folders_path = annotations_folders_path
        self.contours_folders_path = contours_folders_path
        self.images_folders_path = images_folders_path
        self.translations_folders_path = translations_folders_path
        self.davis_paths = [self.annotations_folders_path,
                            self.contours_folders_path,
                            self.images_folders_path,
                            self.translations_folders_path]
        self.parent_model_path = parent_model_path
        
        # Hyperparameters
        self.layer = layer
        self.k = k
        self.augmentation_count = augmentation_count + 1
        
        # Sequences
        self.skip_sequences = skip_sequences
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.sequences = self.train_sequences + self.val_sequences
        self.sequences.sort()
        
        self.train = train
        
        super(DAVIS2016, self).__init__(root, transform=None, pre_transform=None)
        
    @property
    def raw_file_names(self):
        """List of files in root/raw dir which needs to be found in order to skip the download."""
        
        raw_file_names = ['Annotations', 'Contours', 'Images', 'Translations']
        return raw_file_names
    
    def download(self):
        """Moves raw data from DAVIS 2016 folder to root/raw."""
        
        print('Downloading...')
        
        for raw_file_name, davis_path in zip(self.raw_file_names, self.davis_paths):
            raw_dir_path = os.path.join(self.raw_dir, raw_file_name)
            shutil.copytree(davis_path, raw_dir_path)
         
    @property
    def processed_file_names(self):
        """List of files in root/processed which needs to be found in order to skip the processing."""
        
        processed_file_names = []
        
        # Get path to Annotations
        raw_path_annotations = self.raw_paths[0]
        
        # Iterate through sequences 
        for i, sequence in enumerate(self.sequences):
            
            if i > cfg.DEBUG: break
            
            # Skip sequence if needed
            if (sequence in self.skip_sequences): continue
                
            # Train or val dataset
            if self.train:
                if (sequence in self.val_sequences): continue
            else:
                if (sequence in self.train_sequences): continue
                    
            # Iterate through augmentations:
            for j in range(self.augmentation_count):

                j = str(j)

                # Get path to annotion folder
                annotations_folder_path = os.path.join(raw_path_annotations, sequence, j)
                
                # If augmetation does not exist continue with next
                if not os.path.exists(annotations_folder_path):
                    continue
                
                # Get list of frames
                frames = os.listdir(annotations_folder_path)
                if '.ipynb_checkpoints' in frames:
                    frames.remove('.ipynb_checkpoints')
                frames.sort()

                # Iterate through frames
                for k, frame in enumerate(frames[:-1]):

                    if k > cfg.DEBUG: break
                    #print('\t\t#{}: {}'.format(k, frame))
                    
                    # Skip these sequences as annotations are completely black
                    if (sequence == 'bmx-bumps' and frame == '00059.png'): break
                    if (sequence == 'surf' and frame == '00053.png'): break

                    processed_file_names.append('{}_{}_{}.pt'.format(sequence, j, frame[:5]))
        
        return processed_file_names
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def process(self):
        """Processes raw data and saves it into root/processed.
        
        For each frame, a torch_geometric.data object is created that consists of:
        * x: Node feature matrix of shape [num_nodes, num_node_features]. The feature 
             of each node are the concatenated OSVOS feature vectors of the current 
             and the next frame.
        * edge_index: Graph connectivity in COO format of shape (2, num_edges) and type torch.long
                      Each node should be connected to its K nearest neighbours
        * edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                     The feature of each edge is the inverse distance between the two nodes it connects
        * y: The target of each node is the displacement of the node between the current and the next frame
        """
        
        # Get paths to Annotations, Contours, Images, and Translations
        raw_path_annotations, raw_path_contours, raw_path_images, raw_path_translations = self.raw_paths
        
        # Create OSVOS model for feature vector extraction
        print('Create new OSVOS model...')
        new_model = create_osvos_model(self.parent_model_path, self.layer)
        
        # Iterate through sequences 
        for i, sequence in enumerate(self.sequences):
            
            # Debugging
            if i > cfg.DEBUG: break
            
            # Skip sequence if needed
            if (sequence in self.skip_sequences): continue
                    
            # Train or val dataset
            if self.train:
                if (sequence in self.val_sequences): continue
            else:
                if (sequence in self.train_sequences): continue
            
            print('#{}: {}'.format(i, sequence))
                       
            # Iterate through augmentations:
            for j in range(self.augmentation_count):

                j = str(j)

                print('\t{} #{}'.format('Augmentation', j))   
            
                # Get paths to current sequence in Contours, Images, and Translations folders
                contours_folder_path = os.path.join(raw_path_contours, sequence, j)
                images_folder_path = os.path.join(raw_path_images, sequence, j)
                translations_folder_path = os.path.join(raw_path_translations, sequence, j)

                # Get list of Images (one for each frame in the sequence)
                if not os.path.exists(images_folder_path):
                    continue
                frames = os.listdir(images_folder_path)
                if '.ipynb_checkpoints' in frames:
                    frames.remove('.ipynb_checkpoints')
                frames.sort()

                # Iterate through frames
                for k, frame in enumerate(frames[:-1]):

                    file = os.path.splitext(frame)[0] + '.npy'

                    # Debugging
                    if k > cfg.DEBUG: break
                    #print('\t\t#{}: {}'.format(k, frame))
                    
                    # Skip these sequences as annotations are completely black
                    if (sequence == 'bmx-bumps' and frame == '00059.jpg'): break
                    if (sequence == 'surf' and frame == '00053.jpg'): break

                    # Load corresponding contour
                    contour_path = os.path.join(contours_folder_path, file)
                    contour = np.load(contour_path)
                    contour = np.squeeze(contour)

                    # Load corresponding translation
                    translation_path = os.path.join(translations_folder_path, file)
                    translation = np.load(translation_path)

                    # Get image path to current and next frame
                    image_path_0 = os.path.join(images_folder_path, frames[k][:5] + '.jpg')
                    image_path_1 = os.path.join(images_folder_path, frames[k+1][:5] + '.jpg')

                    # Get data
                    data_name = '{}_{}_{}.pt'.format(sequence, j, frame[:5])
                    data_path = os.path.join(self.processed_dir, data_name)
                    if os.path.exists(data_path):
                        continue

                    data = create_data(contour, translation, image_path_0, image_path_1, new_model, self.k)

                    # Save data
                    torch.save(data, data_path)

    def get(self, idx):
        """Load a single data object at given index."""
        
        file_name = self.processed_file_names[idx]
        data_path = os.path.join(self.processed_dir, file_name)
        data = torch.load(data_path)
        return data
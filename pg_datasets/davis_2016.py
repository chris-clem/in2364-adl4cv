from distutils.dir_util import copy_tree
import numpy as np
import os

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

from pg_datasets.create_data import create_data


class DAVIS2016(InMemoryDataset):
    def __init__(self, root, contours_folders_path, translations_folders_path, k, 
                 skip_sequences, train_sequences, val_sequences,
                 train=True, transform=None, pre_transform=None):
        
        self.contours_folders_path = contours_folders_path
        self.translations_folders_path = translations_folders_path 
        self.k = k
        self.skip_sequences = skip_sequences
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        
        super(DAVIS2016, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        
    @property
    def raw_file_names(self):
        raw_file_names = ['Contours', 'Translations']
        return raw_file_names

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt']

    def download(self):
        # Copy Contours folder to raw_dir
        raw_dir_contours = os.path.join(self.raw_dir, 'Contours')
        copy_tree(self.contours_folders_path, raw_dir_contours)
        
        # Copy Translations folder to raw_dir
        raw_dir_translations = os.path.join(self.raw_dir, 'Translations')
        copy_tree(self.translations_folders_path, raw_dir_translations)
        
    def process(self):
        # Get paths to Contours and Translations
        raw_path_contours, raw_path_translations = self.raw_paths
        
        # Get list of folders (there is one for each sequence)
        translations_folders_list = os.listdir(raw_path_translations)
        
        # Create empty data lists to which Data objects will be added
        train_data_list = []
        val_data_list = []
        
        # Iterate through folders 
        for i, folder in enumerate(translations_folders_list):
            
            # Skip if it is a bad sequence
            if (folder in self.skip_sequences): continue
            
            print('#{}: {}'.format(i, folder))
            
            # Get paths to current sequence in Contours and Translations folders
            contours_folder_path = os.path.join(raw_path_contours, folder)
            translations_folder_path = os.path.join(raw_path_translations, folder)
            
            # Get list of translations (one for each frame in the sequence)
            translations = os.listdir(translations_folder_path)
            translations.sort()
            
            # Iterate through translations
            for j, translation in enumerate(translations):
                
                # Load corresponding contour
                contour_path = os.path.join(contours_folder_path, translation)
                contour = np.load(contour_path)
                
                # Load corresponding sequence
                translation_path = os.path.join(translations_folder_path, translation)
                translation = np.load(translation_path)
                
                # Get data and append it to corresponding data_list
                data = create_data(contour, translation, self.k)

                if folder in self.train_sequences:
                    train_data_list.append(data)
                else:
                    val_data_list.append(data)
                
        if self.pre_filter is not None:
            train_data_list = [data for data in train_data_list if self.pre_filter(data)]
            val_data_list = [data for data in val_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            train_data_list = [self.pre_transform(data) for data in train_data_list]
            val_data_list = [self.pre_transform(data) for data in val_data_list]

        for i, data_list in enumerate([train_data_list, val_data_list]): 
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

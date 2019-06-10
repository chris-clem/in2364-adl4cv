#!/usr/bin/env python
# coding: utf-8

from distutils.dir_util import copy_tree
import numpy as np
import os

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


CONTOURS_FOLDERS_PATH = 'DAVIS_2016/DAVIS/Contours/480p'
TRANSLATIONS_FOLDERS_PATH = 'DAVIS_2016/DAVIS/Translations/480p'
PYTORCH_GEOMETRIC_DAVIS_2016_DATASET_PATH = 'PyTorch_Geometric_Datasets/DAVIS_2016'

SKIP_SEQUENCES = ['bmx-trees', 'bus', 'cows', 'dog-agility', 'horsejump-high', 
                  'horsejump-low', 'kite-walk', 'lucia', 'libby', 'motorbike',
                  'paragliding', 'rhino', 'scooter-gray', 'swing']
K = 32


def get_edge_attribute(contour, edge_index):
    '''Returns an edge feature matrix of shape [num_edges, num_edge_features]
       containing the distances between the node each edge connects.'''
    
    edge_index = edge_index.numpy()
    edge_index = edge_index.T
    
    edge_attr = []
    for edge in edge_index:
        contour_point_0 = contour[edge[0]] 
        contour_point_1 = contour[edge[1]]
        dist = np.linalg.norm(contour_point_0-contour_point_1)
        edge_attr.append([dist])
    
    edge_atrr = np.array(edge_attr)
    return torch.from_numpy(edge_atrr)


def create_data(contour, translation):
    '''Returns data object.'''
    
    # x: Node feature matrix with shape [num_nodes, num_node_features]
    # The feature of each node is the OSVOS feature vector of the next frame
    # TODO 
    # x = get_OSVOS_feature_vectors(contour)
    x = None

    # edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    # Each node should be connected to its K nearest neighbours
    positions = torch.from_numpy(contour)
    edge_index = knn_graph(positions, K)
    edge_index = to_undirected(edge_index)

    # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    # The feature of each edge is the distance between the two nodes it connects
    edge_attr = get_edge_attribute(contour, edge_index)

    # y: Target to train against (may have arbitrary shape)
    # The target of each node is the displacement of the node between the current and the next frame
    y = torch.from_numpy(translation)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


class DAVIS2016(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DAVIS2016, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        raw_file_names = ['Contours', 'Translations']
        return raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Copy Contours folder to raw_dir
        raw_dir_contours = os.path.join(self.raw_dir, 'Contours')
        copy_tree(CONTOURS_FOLDERS_PATH, raw_dir_contours)
        
        # Copy Translations folder to raw_dir
        raw_dir_translations = os.path.join(self.raw_dir, 'Translations')
        copy_tree(TRANSLATIONS_FOLDERS_PATH, raw_dir_translations)
        
    def process(self):
        # Get paths to Contours and Translations
        raw_path_contours, raw_path_translations = self.raw_paths
        
        # Get list of folders (there is one for each sequence)
        translations_folders_list = os.listdir(raw_path_translations)
        
        # Create empty data list to which Data objects will be added
        data_list = []
        
        # Iterate through folders 
        for i, folder in enumerate(translations_folders_list):
            
            # Skip if it is a bad sequence
            if (folder in SKIP_SEQUENCES): continue
            
            # Debug
            # if (i > 2): break
            
            print('#{}: {}'.format(i, folder))
            
            # Get paths to current sequence in Contours and Translations folders
            contours_folder_path = os.path.join(raw_path_contours, folder)
            translations_folder_path = os.path.join(raw_path_translations, folder)
            
            # Get list of translations (one for each frame in the sequence)
            translations = os.listdir(translations_folder_path)
            translations.sort()
            
            # Iterate through translations
            for j, translation in enumerate(translations):
                
                # Debug
                # if (j > 4): break
                
                # print('\t#{}: {}'.format(j, translation))
                
                # Load corresponding contour
                contour_path = os.path.join(contours_folder_path, translation)
                contour = np.load(contour_path)
                
                # Load corresponding sequence
                translation_path = os.path.join(translations_folder_path, translation)
                translation = np.load(translation_path)
                
                # Get data and append it to data_list
                data = create_data(contour, translation)
                data_list.append(data)
                
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

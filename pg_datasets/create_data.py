import numpy as np
from scipy.misc import imread, imsave, imresize

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

import OSVOS_PyTorch.networks.vgg_osvos as vo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_OSVOS_feature_vectors(key_point_positions, img, new_model):
    """
    Function takes list of keypoints as input and outputs OSVOS feature vector for each point
    key point positions list of tuples [(x, y), ...]

    """

    #1st CONV block 1-4 --> Shape: [1, 64, 480, 854] --> Receptive field: 5
    #2nd CONV block 5-9 --> Shape: [1, 128, 240, 427] --> Receptive field: 5 * 2(pooling) + 4 = 14
    #3rd CONV block 10-16 --> Shape: [1, 256, 120, 214] --> Receptive field: 14 * 2 + 6 = 34
    #4th CONV block 17-23 --> Shape: [1, 512, 60, 107] --> Receptive field: 34 * 2 + 6 = 74
    #5th CONV block 24-30 --> Shape: [1, 512, 30, 54] --> Receptive field: 74 * 2 + 6 = 154
    
    gpu_id = 0
    img = img.to(device)
    
    with torch.no_grad():
        feature_vector = new_model(img)
    #print('Feature vector shape:', feature_vector.shape)

    #Extract vector out of output tensor depending on keypoint location
    #Compute receptive fields for every feature vector. --> Select feature vector which receptive field center is closest to key point
    _, _, height_img, width_img = img.cpu().numpy().shape
    _, _, height_fv, width_fv = feature_vector.detach().cpu().numpy().shape
    
    feature_vectors = []
    for key_point_position in key_point_positions:
	    x_kp, y_kp = key_point_position
	    x_fv, y_fv = round(float(x_kp) * width_fv / width_img) - 1, round(float(y_kp) * height_fv / height_img) - 1
	    feature_vectors.append(feature_vector[: ,: , y_fv, x_fv].cpu().numpy())

    feature_vectors = np.squeeze(np.array(feature_vectors))
    return torch.from_numpy(feature_vectors)


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
    
    edge_attr = np.array(edge_attr).astype(np.float64)
    edge_attr = np.squeeze(edge_attr)
    
    return torch.from_numpy(edge_attr)


def create_data(contour, translation, img_path, new_model, k):
    '''Returns data object.'''
    
    contour = torch.from_numpy(contour)
    
    img = np.moveaxis(imread(img_path), 2, 0).astype(np.float64)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    
    # x: Node feature matrix with shape [num_nodes, num_node_features]
    # The feature of each node is the OSVOS feature vector of the next frame
    # TODO 
    # x = get_OSVOS_feature_vectors(contour)
    x = get_OSVOS_feature_vectors(contour, img, new_model)

    # edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    # Each node should be connected to its K nearest neighbours
    edge_index = knn_graph(contour, k)
    edge_index = to_undirected(edge_index)

    # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    # The feature of each edge is the distance between the two nodes it connects
    edge_attr = get_edge_attribute(contour, edge_index)
    
    # y: Target to train against (may have arbitrary shape)
    # The target of each node is the displacement of the node between the current and the next frame
    y = torch.from_numpy(translation.astype(np.float64))

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, img=img, contour=contour)
    
    return data
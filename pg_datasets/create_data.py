import numpy as np
from scipy.misc import imread, imsave, imresize

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

import OSVOS_PyTorch.networks.vgg_osvos as vo


def get_OSVOS_feature_vectors(key_point_positions, layer, img_path, model_path):
    """
    Function takes list of keypoints as input and outputs OSVOS feature vector for each point
    key point positions list of tuples [(x, y), ...]

    """

    #1st CONV block 1-4 --> Shape: [1, 64, 480, 854] --> Receptive field: 5
    #2nd CONV block 5-9 --> Shape: [1, 128, 240, 427] --> Receptive field: 5 * 2(pooling) + 4 = 14
    #3rd CONV block 10-16 --> Shape: [1, 256, 120, 214] --> Receptive field: 14 * 2 + 6 = 34
    #4th CONV block 17-23 --> Shape: [1, 512, 60, 107] --> Receptive field: 34 * 2 + 6 = 74
    #5th CONV block 24-30 --> Shape: [1, 512, 30, 54] --> Receptive field: 74 * 2 + 6 = 154

    #Load test image and transform it in (C, H, W)
    img = np.moveaxis(imread(img_path), 2, 0).astype(float)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    print('Image shape:', img.numpy().shape)

    #Load model
    model = vo.OSVOS(pretrained=0)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

    #Define layer as output
    #print('\nSTAGES\n\n', len(model.stages), model.stages)
    #print('\nUPSCALE\n\n', len(model.upscale), model.upscale)
    #print('\nside_prep\n\n', len(model.side_prep), model.side_prep)
    #print('\nscore_dsn\n\n', len(model.score_dsn), model.score_dsn)
    #print('\nupscale_\n\n', len(model.upscale_), model.upscale_)
    #print('\nfuse\n\n', model.fuse)

    children = []
    for num, stage in enumerate(model.stages):
        #print(type(stage), '\n', stage, '\n\n')
        if type(stage) == torch.nn.modules.container.Sequential:
        	for child in stage.children():
        		children.append(child)

    print('Create model for feature_vector after', len(children[:layer]), 'layers...')
    new_model = nn.Sequential(*children[:layer])
    new_model = new_model.double()
    new_model.eval()

    feature_vector = new_model(img)
    print('Feature vector shape:', feature_vector.shape)

    #Extract vector out of output tensor depending on keypoint location
    #Compute receptive fields for every feature vector. --> Select feature vector which receptive field center is closest to key point
    _, _, height_img, width_img = img.numpy().shape
    _, _, height_fv, width_fv = feature_vector.detach().numpy().shape

    feature_vectors = []
    for key_point_position in key_point_positions:
	    x_kp, y_kp = key_point_position
	    x_fv, y_fv = round(float(x_kp) * width_fv / width_img), round(float(y_kp) * height_fv / height_img)
	    #print('X', x_kp, '-->', x_fv, '\nY',  y_kp, '-->', y_fv)
	    feature_vectors.append(feature_vector[: ,: , y_fv, x_fv])

    return feature_vectors


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


def create_data(contour, translation, layer, img_path, model_path, k):
    '''Returns data object.'''
    
    # x: Node feature matrix with shape [num_nodes, num_node_features]
    # The feature of each node is the OSVOS feature vector of the next frame
    # TODO 
    # x = get_OSVOS_feature_vectors(contour)
    x = get_OSVOS_feature_vectors(contour, layer, img_path, model_path)

    # edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    # Each node should be connected to its K nearest neighbours
    positions = torch.from_numpy(contour)
    edge_index = knn_graph(positions, k)
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
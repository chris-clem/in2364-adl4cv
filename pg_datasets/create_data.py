import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


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


def create_data(contour, translation, k):
    '''Returns data object.'''
    
    # x: Node feature matrix with shape [num_nodes, num_node_features]
    # The feature of each node is the OSVOS feature vector of the next frame
    # TODO 
    # x = get_OSVOS_feature_vectors(contour)
    x = None

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
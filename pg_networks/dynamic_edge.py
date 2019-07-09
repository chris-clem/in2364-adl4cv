"""Class for a PyTorch Geometric DynamicEdgeConv network from the paper
Dynamic Graph CNN for Learning on Point Clouds
https://arxiv.org/abs/1801.07829
"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv


# from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))])


# from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/edge_conv.py
class DynamicEdgeConv(EdgeConv):
    def __init__(self, nn, k, aggr='max', **kwargs):
        super(DynamicEdgeConv, self).__init__(nn=nn, aggr=aggr, **kwargs)
        self.k = k
        self.flow = 'source_to_target'
        
    def forward(self, x, batch=None):
        """"""
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn, self.k)
    
    
class DynamicEdge(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=32, aggr='max'):
        super(DynamicEdge, self).__init__()
        
        self.conv1 = DynamicEdgeConv(MLP([in_channels, in_channels * 2]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([in_channels * 2, in_channels * 2]), k, aggr)
        
        self.lin1 = Linear(in_channels * 2, in_channels)
        self.lin2 = Linear(in_channels, out_channels)

    def forward(self, data):
        x = data.x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x)
        
        return x 
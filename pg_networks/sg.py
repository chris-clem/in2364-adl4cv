import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import SGConv

class SG(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SG, self).__init__()
        
        self.conv1 = SGConv(in_channels, in_channels * 2)
        self.conv2 = SGConv(in_channels * 2, in_channels * 2)
        self.conv3 = SGConv(in_channels * 2, in_channels * 4)
        
        self.lin1 = Linear(in_channels * 4, in_channels * 2)
        self.lin2 = Linear(in_channels * 2, in_channels)
        self.lin3 = Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
                
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)        

        x = self.lin3(x)
        
        return x 
import torch
import torch.nn as nn
from layers.graph_operation_layer import GCNLayer


class Graph_Conv_Block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, residual=True):
        super(Graph_Conv_Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.residual = residual
        
        self.gcn1 = GCNLayer(input_dim, output_dim, activation='relu', residual=residual)
        self.gcn2 = GCNLayer(output_dim, output_dim, activation='relu', residual=residual)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                output_dim,
                output_dim,
                (3, 1),
                (1, 1),
                (1, 0),
            ),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(dropout, inplace=False),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (input_dim == output_dim):
            self.residual = lambda x: x
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=False)

    def reshape_to_conv(self, features):
        # (N, T, V, C) -> (N, C, T, V)
        now_feat = features.permute(0, 3, 1, 2).contiguous()
        return now_feat

    def forward(self, graph, x):
        res = self.residual(self.reshape_to_conv(x))
        x = self.gcn1(graph, x)
        x = self.gcn2(graph, x)
        x = self.reshape_to_conv(x)
        x = self.tcn(x) + res
        return self.relu(x)
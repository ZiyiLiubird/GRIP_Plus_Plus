import math

import torch
import torch.nn as nn


def identity_act(input):
    return input

def get_activation(act: str, inplace=False):
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "gelu":
        return nn.GELU()
    elif act == "prelu":
        return nn.PReLU()
    elif act == "identity":
        return identity_act
    else:
        return identity_act


class GCNLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int, dropout=0.0, activation=None, residual=False,):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features == out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        
        if activation is not None:
            self.act = get_activation(activation, inplace=True)
        else:
            self.act = None

    def forward(self, graph, x):
        support = self.linear(x)
        out = torch.matmul(graph, support)
        
        if self.act is not None:
            out = self.act(out)

        if self.residual is not None:
            out = out + self.residual(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN_pytorch.GraphConvolution import GraphConvolution

class GCN(nn.Module):
    def __init__(self,in_size,hidd_size,n_class,dropout):
        super(GCN, self).__init__()
        self.in_size = in_size
        self.hidd_size = hidd_size
        self.n_class = n_class
        self.dropout = dropout
        self.layer = nn.ModuleList()

        self.layer.append(GraphConvolution(self.in_size,self.hidd_size[0]))
        for i in range(len(hidd_size)-1):
            self.layer.append(GraphConvolution(self.hidd_size[i], self.hidd_size[i+1]))
        self.layer.append(GraphConvolution(self.hidd_size[-1],self.n_class))


    def forward(self,input,adj):
        x=F.relu(input)
        for layer in self.layer:
            x = layer(x,adj)
            x = F.relu(x)
            x = F.dropout(x,self.dropout,training=self.training)

        return F.log_softmax(x,dim=1)
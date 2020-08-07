import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import  math

class GraphConvolution(nn.Module):
    def __init__(self,in_size,out_size):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(self.in_size,self.out_size))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv,stdv)

    def forward(self,input,adj):
        support = torch.mm(input,self.weight)
        output = torch.sparse.mm(adj,support)
        return output

    def __repr__(self):
        return self.__class__.__name__+' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'
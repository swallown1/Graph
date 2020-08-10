
import sys
import os

from collections import defaultdict
import numpy as np


class Dataset(object):
    def __init__(self,dataset,path):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.path = path+dataset

    def load_data(self,):
        cora_content_file = self.path + '/cora.content'
        cora_cite_file = self.path + '/cora.cites'

        feat_data = []
        labels = []  # label sequence of node
        node_map = {}  # map node to Node_ID
        label_map = {}  # map label to Label_ID

        with open(cora_content_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels.append(label_map[info[-1]])
        feat_data = np.asarray(feat_data)
        labels = np.asarray(labels, dtype=np.int64)

        adj_lists = defaultdict(set)
        with open(cora_cite_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)

        assert len(feat_data) == len(labels) == len(adj_lists)
        test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

        setattr(self, self.dataset + '_test', test_indexs)
        setattr(self, self.dataset + '_val', val_indexs)
        setattr(self, self.dataset + '_train', train_indexs)

        setattr(self, self.dataset + '_feats', feat_data)
        setattr(self, self.dataset + '_labels', labels)
        setattr(self, self.dataset + '_adj_lists', adj_lists)

    def _split_data(self,num_nodes,test_split=3,val_split = 6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_idx = rand_indices[:test_size]
        val_idx = rand_indices[test_size:(test_size+val_size)]
        train_idx = rand_indices[(test_size+val_size):]

        return  test_idx,val_idx,train_idx
import sys
import os
import torch
import argparse
import random
import numpy as np
from GraphSAGE.utils import *
from GraphSAGE.load_data import Dataset
from GraphSAGE.GraphSAGE import GraphSAGE,Classification,UnsupervisedLoss


parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda")
print('DEVICE:', device)


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load data
    ds = args.dataSet
    data = Dataset(dataset='cora',path='./')
    data.load_data()

    feat = torch.FloatTensor(data.cora_feats).to(device)
    num_labels = len(set(getattr(data, ds + '_labels')))

    graphsage = GraphSAGE([64,64],feat.size(1),128,feat, getattr(data, ds+'_adj_lists'),
                          device, gcn=args.gcn, agg_func=args.agg_func).to(device)

    classification = Classification(128,num_labels).to(device)

    unsupervised_loss = UnsupervisedLoss(getattr(data, ds + '_adj_lists'), getattr(data, ds + '_train'),
                                         device)

    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, classification = apply_model(data, ds, graphSage, classification, unsupervised_loss, args.b_sz,
                                                args.unsup_loss, device, args.learn_method)
        if (epoch + 1) % 2 == 0 and args.learn_method == 'unsup':
            classification, args.max_vali_f1 = train_classification(data, graphSage, classification, ds, device,
                                                                    args.max_vali_f1, args.name)
        if args.learn_method != 'unsup':
            args.max_vali_f1 = evaluate(data, ds, graphSage, classification, device, args.max_vali_f1, args.name,
                                        epoch)

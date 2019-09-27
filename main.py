import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from dataset import dataset
from torch.utils.data import DataLoader
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='')
    parser.add_argument('--gt_dir', type=str, default='')
    
    args = parser.parse_args()

    dataset = dataset(args.result_dir, args.gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    TP = torch.zeros(len(dataloader), 1)
    TN = torch.zeros(len(dataloader), 1)
    P = torch.zeros(len(dataloader), 1)
    N = torch.zeros(len(dataloader), 1)
    for i, (pred, gt) in enumerate(dataloader):
        pred = pred.cuda()
        gt = gt.cuda()
        
        pred_clone = torch.zeros_like(pred).cuda()
        gt_clone = torch.zeros_like(gt).cuda()
        
        thres = 125 / 255.
        mask = pred > thres
        pred_clone[mask] = 1

        gt_clone[gt > 0.5] = 1

        TP[i, 0] = (pred_clone * gt_clone).sum()
        TN[i, 0] = ((1 - pred_clone) * (1 - gt_clone)).sum()
        P[i, 0] = gt_clone.sum()
        N[i, 0] = (1 - gt_clone).sum()
        
    ber = 1 - 0.5 * (TP.sum()/P.sum() + TN.sum()/N.sum())
    p_error = 1 - TP.sum() / P.sum()
    n_error = 1 - TN.sum() / N.sum()
    acc_final = (TP.sum() + TN.sum()) / (P.sum() + N.sum())
 
    print('ber = ', ber*100)
    print('p_error = ', p_error*100)
    print('n_error = ', n_error*100)
    print('acc_final = ', acc_final*100)

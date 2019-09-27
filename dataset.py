import torch
import numpy as np
from torch.utils import data
from torchvision import transforms

from PIL import Image
import os

class dataset(data.Dataset):
    def __init__(self, result_dir, gt_dir):
        self.pred = []
        self.gt = []

        list_dir = os.listdir(result_dir)
        for name in list_dir:
            self.pred.append(result_dir + name)
            self.gt.append(gt_dir + name[:-4] + '.png')

    def __getitem__(self, item):
        #print(self.pred[item], self.gt[item])
        pred = Image.open(self.pred[item])
        gt = Image.open(self.gt[item])
        totensor_op = transforms.ToTensor()
        pred = totensor_op(pred)
        gt = totensor_op(gt)
        return pred, gt
    def __len__(self):
        return len(self.pred)

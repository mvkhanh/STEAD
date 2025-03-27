import numpy as np
import torch
from torch import nn
import option
import cv2
from PIL import Image

# args=option.parse_args()


def save_best_record(test_info, file_path):
    f = open(file_path, "w")
    f.write("epoch: {}\n".format(test_info["epoch"][-1]))
    f.write(str(test_info["test_AUC"][-1]))
    f.write("\n")
    f.write(str(test_info["test_PR"][-1]))
    f.close()

def FeedForward(dim, repe = 4, dropout=0.):
    return nn.Sequential(
        nn.Linear(dim, dim * repe),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * repe, dim),
        nn.GELU(),
    )

class DECOUPLED(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        kernel = 3
    ):
        super().__init__()
        self.heads = heads
        self.norm2d = nn.BatchNorm2d(dim)
        self.norm1d = nn.BatchNorm1d(dim)
        self.conv2d = nn.Conv2d(dim, dim, kernel, padding = kernel // 2, groups = heads)
        self.conv1d = nn.Conv1d(dim, dim, kernel, padding = kernel // 2, groups = heads)


    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, C, H, W)
        x = self.norm2d(x)
        x = self.conv2d(x)
        x = x.view(B * H * W, C, T)
        x = self.norm1d(x)
        x = self.conv1d(x)
        x = x.view(B, T, H, W, C)
        return x


import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import random as rand


def _fast_hist(label_true, label_pred, n_class):

    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

"""
label_trues: right label
label_preds: predic label
n_class: number of classes
"""
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        # numpy.ndarray.flatten : numpy to 1D
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    
    # np.diag(a)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)  # nanmean ignore NaN
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc    

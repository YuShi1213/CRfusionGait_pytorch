from config import *
import random
import math
import torch
import numpy as np


def collate_fn(batch):

    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    spes = [batch[i][1] for i in range(batch_size)]
    label = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    view = [batch[i][4] for i in range(batch_size)]
    batch = [seqs, spes,label, seq_type, view]
    seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    spes = [np.asarray([spes[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    batch[0] = seqs
    batch[1] = spes
    return batch

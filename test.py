from datetime import datetime
import numpy as np
import argparse
from config import *
import random
import torch
import math
from model import SetNet
from data import load_data
from utils import collate_fn
import torch.autograd as autograd
import torch.nn.functional as F

test_batch_size = 8

def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 7.0
    if not each_angle:
        result = np.mean(result)
    return result

def test():

    model = SetNet(hidden_dim=hidden_dim).float()
    model.load_state_dict(torch.load(os.path.join(
        'checkpoint', model_name,
        '{}-{:0>5}-encoder-gei.ptm'.format(save_name, test_restore_iter))))
    model.cuda()
    model.eval()
    print('Complete loading model!')

    test_source = load_data('test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_source,
        batch_size=test_batch_size,
        sampler = torch.utils.data.sampler.SequentialSampler(test_source),
        collate_fn = collate_fn,
        num_workers = 8
    )

    feature_list = []
    label_list = []
    condition_list = []
    angle_list = []
    for _, (seq,spe,identity,condition,angle) in enumerate(test_loader):
        for i in range(len(seq)):
            seq[i] = autograd.Variable(torch.from_numpy(seq[i])).cuda().float()
        for i in range(len(spe)):
            spe[i] = autograd.Variable(torch.from_numpy(spe[i])).cuda().float()
        feature = model(*seq,*spe)
        a,b,c = feature.size()
        feature_list.append(feature.view(a,-1).data.cpu().numpy())
        label_list += identity
        condition_list += condition
        angle_list += angle

    feature_list = np.concatenate(feature_list,0)
    label_list = np.array(label_list)
    num_angle = len(set(angle_list))
    angle_set_list = list(set(angle_list))

    print('Finish Loading data!')
    acc_table = np.zeros([len(test_probe_condition_list),num_angle,num_angle])
    for (con,probe_condition) in enumerate(test_probe_condition_list):
        # for gallery_condition in test_gallery_condition_list:
        for (a1, probe_angle) in enumerate(sorted(angle_set_list)):
            for (a2,gallery_angle) in enumerate(sorted(angle_set_list)):

                gallery_mask = np.isin(condition_list,test_gallery_condition_list) & np.isin(angle_list,[gallery_angle])
                gallery_feature = feature_list[gallery_mask,:]
                gallery_label = label_list[gallery_mask]

                probe_mask = np.isin(condition_list,probe_condition) & \
                             np.isin(angle_list,[probe_angle])
                probe_feature = feature_list[probe_mask,:]
                probe_label = label_list[probe_mask]

                dist = cuda_dist(probe_feature,gallery_feature)
                a = dist
                idx = dist.sort(1)[1].cpu().numpy()

                acc_table[con,a1,a2] = np.round(
                    np.sum((probe_label == gallery_label[idx[:,0]])>0,\
                           0)*100 / dist.shape[0],2)


    print('===Rank-%d (Include identical-view cases)===' % (1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(acc_table[0, :, :]),
        np.mean(acc_table[1, :, :]),
        np.mean(acc_table[2, :, :])))

    print('===Rank-%d (Exclude identical-view cases)===' % (1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        de_diag(acc_table[0, :, :]),
        de_diag(acc_table[1, :, :]),
        de_diag(acc_table[2, :, :])))

    np.set_printoptions(precision=2, floatmode='fixed')
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (1))
    print('NM:', de_diag(acc_table[0, :, :], True))
    print('BG:', de_diag(acc_table[1, :, :], True))
    print('CL:', de_diag(acc_table[2, :, :], True))
    acc_table2 = np.zeros([len(test_probe_condition_list),num_angle])
    for (con,probe_condition) in enumerate(test_probe_condition_list):
        for gallery_condition in test_gallery_condition_list:
            for (a1, probe_angle) in enumerate(sorted(angle_set_list)):
                    gallery_mask = np.isin(condition_list,gallery_condition)
                    gallery_feature = feature_list[gallery_mask,:]
                    gallery_label = label_list[gallery_mask]  #

                    probe_mask = np.isin(condition_list,probe_condition) & \
                                 np.isin(angle_list,[probe_angle])
                    probe_feature = feature_list[probe_mask,:]
                    probe_label = label_list[probe_mask]
                    dist = cuda_dist(probe_feature,gallery_feature)
                    idx = dist.sort(1)[1].cpu().numpy()

                    acc_table2[con,a1] = np.round(
                        np.sum((probe_label == gallery_label[idx[:,0]])>0,\
                               0)*100 / dist.shape[0],2)
    print('===Rank-%d (Multiple-view cases)===' % (1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(acc_table2[0, :]),
        np.mean(acc_table2[1, :]),
        np.mean(acc_table2[2, :])))
    np.set_printoptions(precision=2, floatmode='fixed')
    print('===Rank-%d of each angle (Multiple-view cases)===' % (1))
    print('NM:', acc_table2[0, :])
    print('BG:', acc_table2[1, :])
    print('CL:', acc_table2[2, :])

if __name__ == '__main__':
    test()

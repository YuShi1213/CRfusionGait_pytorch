from config import *
from data import load_data
from sampler import TripletSampler
import torch
import numpy as np
import random
import math
import torch.autograd as autograd
import torch.optim as optim
from model import SetNet
from tripletloss import TripletLoss
from datetime import datetime
import sys
from utils import collate_fn
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn.functional as F

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
def save(model,optimizer,iteration):

    os.makedirs(os.path.join('checkpoint', model_name), exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join('checkpoint', model_name,
                        '{}-{:0>5}-encoder-gei.ptm'.format(
                            save_name, iteration)))
    torch.save(optimizer.state_dict(),
               os.path.join('checkpoint', model_name,
                        '{}-{:0>5}-optimizer-gei.ptm'.format(
                            save_name, iteration)))

# restore_iter: iteration index of the checkpoint to load
def load(iteration,model,optimizer):

    model.load_state_dict(torch.load(os.path.join(
        'checkpoint', model_name,
        '{}-{:0>5}-encoder-gei.ptm'.format(save_name, iteration))))
    optimizer.load_state_dict(torch.load(os.path.join(
        'checkpoint', model_name,
        '{}-{:0>5}-optimizer-gei.ptm'.format(save_name, iteration))))

def train():

    train_source = load_data(flag='train')
    triplet_sampler = TripletSampler(train_source,batch_size)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_source,
        batch_sampler = triplet_sampler,
        collate_fn = collate_fn,
        num_workers = 8
    )
    test_source = load_data('test')
    test_loader = torch.utils.data.DataLoader(
        dataset = test_source,
        batch_size=1,
        sampler = torch.utils.data.sampler.SequentialSampler(test_source),
        collate_fn = collate_fn,
        num_workers = 8
    )


    model = SetNet(hidden_dim=hidden_dim).float()
    model.cuda()
    model.train()
    num_person,num_sample = batch_size
    Loss =  TripletLoss(num_person*num_sample,margin).cuda()
    optimizer = optim.Adam([{'params':model.parameters()}],lr=lr)


    iteration = train_start_iteration
    log_path = './log.txt'
    if train_start_iteration != 0:
        load(iteration,model,optimizer)
    # else:
    #     with open(log_path, 'w') as f:
    #         f.write("start" + '\n')

    hard_loss_metric_list = []
    full_loss_metric_list = []
    full_loss_num_list = []
    dist_list = []
    # mean_dist = 0.01
    _time1 = datetime.now()
    for seq,spe,identity,condition,angle in train_loader:
        iteration += 1
        optimizer.zero_grad()
        for i in range(len(seq)):
            seq[i] = autograd.Variable(torch.from_numpy(seq[i])).cuda().float()
        for i in range(len(spe)):
            spe[i] = autograd.Variable(torch.from_numpy(spe[i])).cuda().float()
        label = [train_source.label_set.index(l) for l in identity]
        label = autograd.Variable(torch.from_numpy(np.array(label))).cuda().long()

        b = spe

        feature = model(*seq,*spe)

        triplet_feature = feature.permute(1, 0, 2).contiguous()
        triplet_label = label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
        (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
         ) = Loss(triplet_feature, triplet_label)
        loss = full_loss_metric.mean()

        hard_loss_metric_list.append(hard_loss_metric.mean().data.cpu().numpy())
        full_loss_metric_list.append(full_loss_metric.mean().data.cpu().numpy())
        full_loss_num_list.append(full_loss_num.mean().data.cpu().numpy())
        dist_list.append(mean_dist.mean().data.cpu().numpy())

        if loss > 1e-9:
            loss.backward()
            optimizer.step()

        if iteration % 1000 == 0:
            print(datetime.now() - _time1)
            _time1 = datetime.now()

        if iteration % 100 == 0:
            print('iter {}:'.format(iteration), end='')
            print(', hard_loss_metric={0:.8f}'.format(np.mean(hard_loss_metric_list)), end='')
            print(', full_loss_metric={0:.8f}'.format(np.mean(full_loss_metric_list)), end='')
            print(', full_loss_num={0:.8f}'.format(np.mean(full_loss_num_list)), end='')
            with open(log_path, 'a') as f:
                f.write('iter {}:'.format(iteration)+', full_loss_metric={0:.8f}'.format(np.mean(full_loss_metric_list))+'\n')
            mean_dist = np.mean(dist_list)
            print(', mean_dist={0:.8f}'.format(mean_dist), end='')
            print(', lr=%f' % optimizer.param_groups[0]['lr'], end='\n')
            # print(', hard or full=%r' % hard_or_full_trip)
            sys.stdout.flush()
            hard_loss_metric_list = []
            full_loss_metric_list = []
            full_loss_num_list = []
            dist_list = []

        if iteration % 10000 == 0:
            save(model,optimizer,iteration)
        if iteration % 10000 == 0:
            # validate
            model.eval()
            with torch.no_grad():
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

        if iteration == total_iter:
            break


if __name__ == '__main__':
    train()

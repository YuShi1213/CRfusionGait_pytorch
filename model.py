import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from non_local_gaussian import NONLocalBlock1D

####sigmoid_mlp####
def conv1d(in_planes,out_planes,kernal_size,has_bias=False,**kwargs):
    return nn.Conv1d(in_planes, out_planes,kernal_size,bias=has_bias,**kwargs)
def mlp_sigmoid(in_planes,out_planes,kernel_size,**kwargs):
    return nn.Sequential(conv1d(in_planes,in_planes//16,kernel_size,**kwargs),
                        nn.BatchNorm1d(in_planes//16),
                        nn.LeakyReLU(inplace=True),
                        conv1d(in_planes//16,out_planes,kernel_size,**kwargs),
                        nn.Sigmoid())
class part_score(nn.Module):
    def __init__(self,channel,reduction=2):
        super(part_score,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel//reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,p=x.size()
        x1 = x.permute(0,2,1)
        y = self.avg_pool(x1)
        y = self.fc(y).permute(0,2,1)
        return x+x*y.expand_as(x)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):

    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_size, p_stride,padding,do_pool):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding)
        self.norm = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.do_pool = do_pool
        self.pool = nn.MaxPool2d(kernel_size=p_size, stride=p_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.do_pool == True:
            x = self.pool(x)
        return x

class SetNet(nn.Module):

    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 7, padding=3))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 7, padding=3), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 5, padding=2))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 5, padding=2), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))
#################################################################  radar spe ###################################################
        self.channels = [3,32,64,128]
        self.ks = [5,3]
        self.p_size = 2
        self.p_stride = 2
        self.conv1 = Conv2d(self.channels[0],self.channels[1],self.ks[0],self.p_size,self.p_stride,padding=2,do_pool=False)
        self.conv2 = Conv2d(self.channels[1],self.channels[1],self.ks[0],self.p_size,self.p_stride,padding=2,do_pool=True)
        self.conv3 = Conv2d(self.channels[1],self.channels[2],self.ks[1],self.p_size,self.p_stride,padding=1,do_pool=False)
        self.conv4 = Conv2d(self.channels[2],self.channels[2],self.ks[1],self.p_size,self.p_stride,padding=1,do_pool=True)
        self.conv5 = Conv2d(self.channels[2],self.channels[3],self.ks[1],self.p_size,self.p_stride,padding=1,do_pool=False)
        self.conv6 = Conv2d(self.channels[3],self.channels[3],self.ks[1],self.p_size,self.p_stride,padding=0,do_pool=False)
        self.relation = NONLocalBlock1D(128)
        self.score = mlp_sigmoid(128,128,1,groups=1)
        self.score1 = mlp_sigmoid(64,64,1,groups=1)
        self.score2 = mlp_sigmoid(32,32,1,groups=1)
        self.score_gei = mlp_sigmoid(128,1,1,groups=1)
        self.score_gei1 = mlp_sigmoid(64,1,1,groups=1)
        self.score_gei2 = mlp_sigmoid(32,1,1,groups=1)
        self.radar_bin_num = [15]
        self.radar_bin_num1 = [16]
        self.radar_fc_bin = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum([16]), 128, hidden_dim))
        )
        self.radar_fc_bin1 = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum([17]), 32, hidden_dim))
        )
        self.radar_fc_bin2 = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum([17]), 64, hidden_dim))
        )
###################################################################################################################################
        self.bin_num = [16]
        self.bin_num1 = [16]

        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum([16]), 128, hidden_dim))
        )
        self.fc_bin1 = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum(self.bin_num1), 32, hidden_dim))
        )
        self.fc_bin2 = nn.Parameter(
            nn.init.xavier_uniform_( torch.zeros(sum(self.bin_num), 64, hidden_dim))
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, x,y):

        x = x.unsqueeze(2)

        x = self.set_layer1(x)
        x = self.set_layer2(x)

        mid1 = self.frame_max(x)[0]

        x = self.set_layer3(x)
        x = self.set_layer4(x)

        mid2 = self.frame_max(x)[0]

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = x

        final_feature = []
        n,c,h,w = gl.size()
        feature = []
        for num_bin in self.bin_num:
            z = x.view(n,c,num_bin,-1)
            z = z.mean(3) + z.max(3)[0]
            #z = self.score_gei(z)
            pred_score = self.score_gei(z)
            #score = pred_score
            long = z.mul(pred_score)
            #z = z+long
            feature.append(long)
        feature = torch.cat(feature,2).permute(2,0,1).contiguous()
        feature = feature.matmul(self.fc_bin)
        feature = feature.permute(1, 0, 2).contiguous()
        final_feature.append(feature)

        n,c,h,w = mid1.size()
        feature1 = []
        for num_bin in self.bin_num1:
            z = mid1.view(n,c,16,-1)
            z = z.mean(3) + z.max(3)[0]
            pred_score = self.score_gei2(z)
            #score = pred_score.div(pred_score.sum(-1).unsqueeze(1))
            long = z.mul(pred_score)
            #z = z+long
            feature1.append(long)
        feature1 = torch.cat(feature1,2).permute(2,0,1).contiguous()
        feature1 = feature1.matmul(self.fc_bin1)
        feature1 = feature1.permute(1, 0, 2).contiguous()
        final_feature.append(feature1)

        n,c,h,w = mid2.size()
        feature2 = []
        for num_bin in self.bin_num:
            z = mid2.view(n,c,num_bin,-1)
            z = z.mean(3) + z.max(3)[0]
            pred_score = self.score_gei1(z)
            #score = pred_score.div(pred_score.sum(-1).unsqueeze(1))
            long = z.mul(pred_score)
            #z = z+long
            feature2.append(long)
        feature2 = torch.cat(feature2,2).permute(2,0,1).contiguous()
        feature2 = feature2.matmul(self.fc_bin2)
        feature2 = feature2.permute(1, 0, 2).contiguous()
        final_feature.append(feature2)

###########################################################################################################################
        y = self.conv1(y)
        y = self.conv2(y)
        radar_multi1 = y
        y = self.conv3(y)
        y = self.conv4(y)
        radar_multi2 = y
        y = self.conv5(y)
        y = self.conv6(y)

        n,c,h,w = y.size()
        radar_feature = []
        for num_bin in self.radar_bin_num:
            z = y.view(n,c,-1,num_bin)
            z = z.mean(2) + z.max(2)[0]
            pred_score = self.score(z)
            long = z.mul(pred_score).sum(-1).div(pred_score.sum(-1))
            long = long.unsqueeze(2)
            z1 = torch.cat((z,long),2)
            #z = self.relation(z)
            radar_feature.append(z1)
        radar_feature = torch.cat(radar_feature,2).permute(2,0,1).contiguous()
        radar_feature = radar_feature.matmul(self.radar_fc_bin)
        radar_feature = radar_feature.permute(1, 0, 2).contiguous()
        final_feature.append(radar_feature)
        ##multilayer1
        n,c,h,w = radar_multi1.size()
        radar_feature1 = []
        for num_bin in self.radar_bin_num:
            z = radar_multi1.view(n,c,-1,16)
            z = z.mean(2) + z.max(2)[0]
            pred_score = self.score2(z)
            long = z.mul(pred_score).sum(-1).div(pred_score.sum(-1))
            long = long = long.unsqueeze(2)
            z = torch.cat((z,long),2)
            radar_feature1.append(z)
        radar_feature1 = torch.cat(radar_feature1,2).permute(2,0,1).contiguous()
        radar_feature1 = radar_feature1.matmul(self.radar_fc_bin1)
        radar_feature1 = radar_feature1.permute(1, 0, 2).contiguous()
        final_feature.append(radar_feature1)
        ##multilayer2
        n,c,h,w = radar_multi2.size()
        radar_feature2 = []
        for num_bin in self.radar_bin_num:
            z = radar_multi2.view(n,c,-1,16)
            z = z.mean(2) + z.max(2)[0]
            pred_score = self.score1(z)
            long = z.mul(pred_score).sum(-1).div(pred_score.sum(-1))
            long = long = long.unsqueeze(2)
            z = torch.cat((z,long),2)
            radar_feature2.append(z)
        radar_feature2 = torch.cat(radar_feature2,2).permute(2,0,1).contiguous()
        radar_feature2 = radar_feature2.matmul(self.radar_fc_bin2)
        radar_feature2 = radar_feature2.permute(1, 0, 2).contiguous()
        final_feature.append(radar_feature2)

        final_feature = torch.cat(final_feature,1)

        return final_feature

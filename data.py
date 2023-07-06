from torch.utils.data import Dataset
from config import *
import os
import xarray
import numpy as np
import cv2
from PIL import Image

class load_data(Dataset):

    def __init__(self,flag):

        super(load_data, self).__init__()

        self.flag = flag
        if flag == 'train':
            flag_list = train_list
        else:
            flag_list = test_list

        self.identity_list = []
        self.pic_list = []
        self.spe_list = []
        self.condition_list = []
        self.angle_list = []

        for identity in flag_list:
            identity_path = os.path.join(datapath,identity)
            for condition in sorted(os.listdir(identity_path)):
                condition_path = os.path.join(identity_path,condition)
                for angle in sorted(angle_list):
                    radar_identity_path = os.path.join(spe_path,identity)
                    radar_condition_path = os.path.join(radar_identity_path,condition)
                    spe = radar_condition_path+'/'+identity+'-'+condition+'-'+angle+'.png'
                  
                    pic = condition_path+'/'+identity+'-'+condition+'-'+angle+'.png'
                    if os.path.exists(spe) and os.path.exists(pic):
                        self.pic_list.append([pic])
                        self.spe_list.append([spe])
                        self.identity_list.append(identity)
                        self.condition_list.append(condition)
                        self.angle_list.append(angle)

        self.data_size = len(self.identity_list)
        self.label_set = sorted(list(set(self.identity_list)))
        self.condition_set = sorted(list(set(self.condition_list)))
        self.angle_set = sorted(list(set(self.angle_list)))
        _ = np.zeros((len(self.label_set),
                      len(self.condition_set),
                      len(self.angle_set))).astype('int')

        self.index_dict = xarray.DataArray(
            _,
            coords = {'label':self.label_set,
                      'condition':self.condition_set,
                      'angle':self.angle_set},
            dims=['label','condition','angle']
        )

        for i in range(self.data_size):
            label = self.identity_list[i]
            condition = self.condition_list[i]
            angle = self.angle_list[i]
            self.index_dict.loc[label,condition,angle] = i

    def __len__(self):

        return len(self.identity_list)

    def process_img(self,path):

        # imgs = sorted(os.listdir(path))
        frame_list = [np.reshape(
            cv2.imread(path),
            [resolution,resolution,-1])[:,:,0]]
        num_list = list(range(len(frame_list)))
        data_dict = xarray.DataArray(
            frame_list,
            coords={'frame':num_list},
            dims = ['frame','img_y','img_x'],
        )
        cut_array = data_dict[:,:,cut_padding:-cut_padding].astype('float32') / 255.0

        return cut_array

    def process_spe(self,paths):

        path1 = paths[0]
    

        img1 = Image.open(path1)
        img1 = np.array(img1.resize((128, 88), Image.ANTIALIAS)).astype('float32')/255.0
      
        img = np.transpose(img1, (2, 0, 1))

        return img

    def __getitem__(self, item):

        a = self.pic_list[item]

        data = [self.process_img(path) for path in self.pic_list[item]]
        # frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
        # frame_set = list(set.intersection(*frame_set))

        spe_data = [self.process_spe(self.spe_list[item])]

        return data,spe_data,self.identity_list[item],\
               self.condition_list[item],self.angle_list[item]












import os

datapath ="/media/ai/d899633d-2ef9-4e8d-ac69-eb22cae4d04f/gyc/shiyu/gaitcode-set/GEI_spe/data/geiori"
#spe_path ="/media/ai/d899633d-2ef9-4e8d-ac69-eb22cae4d04f/gyc/shiyu/gaitcode-set/GEI_spe/data/spe"
spe_path ="/media/ai/d899633d-2ef9-4e8d-ac69-eb22cae4d04f/gyc/shiyu/GEI_spe/datasets/spectrogram2.4s/"
identity_list = sorted(os.listdir(datapath))
train_list = identity_list[0:74]
test_list = identity_list[74:]
resolution = 128
cut_padding = int(float(resolution)/64*10)
batch_size = (2,4)
# sample_type ='random'
frame_num=10
train_start_iteration = 0
hidden_dim = 256
margin = 0.2
lr = 0.0001
total_iter = 400000
model_name = '6-CNN-avgcut-shared-independent-fc'
save_name = '6-CNN-avgcut-shared-independent-fc'
test_probe_condition_list = [['nm-05', 'nm-06'],['bg-01', 'bg-02'],['ct-01', 'ct-02']]
test_gallery_condition_list = [['nm-01', 'nm-02', 'nm-03', 'nm-04']]
angle_list = ['000','030','045','060','090','300','315','330']
test_restore_iter = 330000


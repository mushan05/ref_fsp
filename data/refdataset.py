import os
import numpy as np
import random
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.utils import *
import torch.distributed as dist
from imageio import imwrite, imread
import torch.nn.functional as F
import cv2

def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image

class R2CObjData(Dataset):

    def __init__(self, data_root, mode='train', shot=5):
        
        print('this is refdataset')
        
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_root = data_root
        self.shot = shot
        
        self.data_list, self.class_file_list = collect_r2c_data(data_root=self.data_root, mode=self.mode)

        if self.mode == 'val' and self.shot not in [-1, 0, 5]:
            # get file_paths
            self.record_class_files()

        # self.img_transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((image_size, image_size)),
        #     transforms.ToTensor()])


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]

        name = image_path.split('/')[-1][:-4]

        # read query imgs/gts
        # image = rgb_loader(image_path)          # PIL格式
        # label = binary_loader(label_path)
        img = imread(image_path).astype(np.float32)/255.
        label = imread(label_path).astype(np.float32)/255.

        # data augumentation
        if self.mode == 'train':
            # image, label = self.aug_data(image=image, label=label)
            ration = np.random.rand()
            if ration<0.25:
                img = cv2.flip(img, 1)
                label = cv2.flip(label, 1)
            elif ration<0.5:
                img = cv2.flip(img, 0)
                label = cv2.flip(label, 0)
            elif ration<0.75:
                img = cv2.flip(img, -1)
                label = cv2.flip(label, -1)
            if len(label.shape)==3:
                label=label[:,:,0]
            label=label[:,:,np.newaxis]
            img = torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0)
            label = torch.from_numpy(label).permute(2,0,1).unsqueeze(0)

        # transforms
        # image = self.img_transform(image)
        # if self.mode == 'train':
        #     label = self.gt_transform(label)
        else:   # val or test
            # label = np.asarray(label, np.float32)    # 保存原始的label,用于测试
            label = label[:,:,np.newaxis]
            img = F.interpolate(torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0), (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0)
            label = torch.from_numpy(label).permute(2,0,1)
            
        
        # read referring represenations
        if self.shot > 0 or self.shot == -1:           
            class_chosen = image_path.split('/')[-1].split('-')[-2]

            file_class_chosen = self.class_file_list[class_chosen]
            num_aux = len(file_class_chosen)
            ref_feat_list = []

            if self.mode == 'train':
                ref_idx_list = random.sample(range(num_aux), self.shot) if self.shot > 0 else list(range(num_aux))   # aux_num > shot
            else:
                ref_idx_list = list(range(num_aux))                         # aux_num == shot
            
            for idx in ref_idx_list:
                ref_feat = np.load(file_class_chosen[idx])
                ref_feat = torch.from_numpy(ref_feat)

                ref_feat_list.append(ref_feat)
                

        if self.shot > 0 or self.shot == -1:
            ref_num = len(ref_feat_list)   
            sal_f = sum(ref_feat_list) / ref_num

        else:   # Baseline
            sal_f = -1                                                                                         


        return img, label, sal_f, name
        
    
    def record_class_files(self):
        '''
        1 <= shot < 5, generating record files
        '''
        file_path = './data/dataset_{}shot_val.json'.format(self.shot)

        if os.path.exists(file_path):
            print('load from {}...'.format(file_path))
            with open(file_path, 'r') as f:
                self.class_file_list = json.load(f)
        else:
            print('generating {}...'.format(file_path))
            for cate in self.class_file_list.keys():
                cate_file_pairs = self.class_file_list[cate]
                assert len(cate_file_pairs) > self.shot
                rand_idxs = random.sample(range(len(cate_file_pairs)), self.shot)
                self.class_file_list[cate] = [cate_file_pairs[idx] for idx in rand_idxs]
            with open(file_path, 'w') as f:
                json.dump(self.class_file_list, f, indent=4)
                
    # def aug_data(self, image, label):
    #     image, label = cv_random_flip(image, label)
    #     image, label = randomCrop(image, label)
    #     image, label = randomRotation(image, label)
    #     image = colorEnhance(image)
    #     label = randomPeper(label)
    #     return image, label

    

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning) 
import os
import random
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision import utils

import albumentations as alb

seed=100 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Deepfake(Dataset):
    def __init__(self, phase='train',csv_dir='default', resize=(224,224)):
        self.phase=phase
        self.resize=resize
        self.type_='c23'

        if csv_dir == 'default':
            csv_dir = os.path.join('./datasets','{}_{}_dlib2.csv'.format(self.type_, phase))

        print('csv_dir:',phase,csv_dir)  

        self.mask_path = ''

        # read img path
        self.data = pd.read_csv(csv_dir)
        self.img_list = list(self.data['file_name'])
        self.img_label = list(self.data['cls'])

        p_color_rgb=0.5 
        p_color_hsv=0.5 
        p_color_bc=0.5 
        p_noise = 0.5
        p_blur = 0.5
        p_compress = 0.5

        self.aug_alb = alb.Compose([
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=p_color_rgb),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=p_color_hsv),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=p_color_bc),
            alb.GaussNoise(p=p_noise),
        
            alb.OneOf([
                alb.MotionBlur(p=0.2),   
                alb.MedianBlur(blur_limit=3, p=0.1),    
                alb.Blur(blur_limit=3, p=0.1),   
            ], p=p_blur),
        
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=p_compress),
            ]) 

        self.trans = T.Compose([
            T.Resize((self.resize[0], self.resize[1])),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.img_label[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        get_mask = True
        if get_mask:
            hard_mask = self.get_mask_area(img_path, self.mask_path, phase=self.phase, mask_pacth_size=self.resize[0])   

        if self.phase == 'train':
            if np.random.rand()<0.5:
                image = image[:,::-1]
                hard_mask = torch.flip(hard_mask, dims=[1])

            image_aug = self.aug_alb(image=image)['image']
            image_aug = Image.fromarray(image_aug)

            return self.trans(image_aug),label, img_path, hard_mask
        else: #val test
            image = Image.fromarray(image)
            return self.trans(image), label, img_path, hard_mask
        
    def collate_fn(self, batch, type_):
        data = {}
        if type_ == 'train':
            images, labels, file_index, hard_mask = tuple(zip(*batch))
            data['img'] = torch.stack(images, dim=0)
            data['label'] = torch.as_tensor(labels)
            data['file_index'] = list(file_index)  
            data['hard_mask'] = torch.stack(hard_mask, dim=0)

        else:
            images, labels, file_index, hard_mask = tuple(zip(*batch))
            data['img'] = torch.stack(images, dim=0)
            data['label'] = torch.as_tensor(labels)
            data['file_index'] = list(file_index)
            data['hard_mask'] = torch.stack(hard_mask, dim=0)

        return data

    def get_mask_area(self, img_path, mask_path, phase='train', mask_pacth_size=224):        
        mask = np.zeros((mask_pacth_size,mask_pacth_size))
        hard_mask = None
        if 'manipulated_sequences' in img_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask = cv2.resize(mask,(mask_pacth_size,mask_pacth_size))
            mask = torch.tensor(mask/255.0)
            hard_mask = (mask>0)*1    
            #cv2.imwrite('./visual/mask/{}'.format(mask_path.split('/')[-2]+'_'+mask_path.split('/')[-1]), hard_mask.numpy()*255)

        return hard_mask


if __name__ == '__main__':
    batch_size=32
    train_dataset = Deepfake(phase='train',
                         csv_dir='default', 
                         resize=(224, 224))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                            collate_fn=lambda x:train_dataset.collate_fn(x,'train'),
                                            shuffle=False,pin_memory=True, num_workers=0)    
    for step, data in enumerate(train_loader):
        images,labels,file_index = \
                data['img'],data['label'],data['file_index']


        images=images.view((-1,3,224,224))
        utils.save_image(images, './data/loader{}.png'.format(step), nrow=batch_size, normalize=False, range=(0, 1))
        print(step,images.shape,labels.shape, len(file_index),'*********')
        break


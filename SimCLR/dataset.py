import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import cv2
import pandas as pd

classes = ['hippedRoof', 'gabledRoof', 'flatRoof', 'monopitchRoof',
           'pyramidalRoof', 'copulaRoof', 'halfHippedRoof', 'archRoof','mansardRoof']


class ClassifDataset(data.Dataset):
    def __init__(self, data_dir,mode = 'df_tagging_roof_material_type',classes = classes, transform=None, recursive_search=True):
        
        def get_df(dirr):    
            df = pd.read_csv(dirr+'/'+mode+'.csv')
            df = df.sample(frac=1).reset_index(drop=True)
            return df 

        super(ClassifDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.df = get_df(self.data_dir)
        self.transform = transform
        self.classes = classes
        

    def __getitem__(self, index, color_format='RGB'):
        def to_categorical(y, num_classes):
            """ 1-hot encodes a tensor """
            return np.eye(num_classes, dtype='uint8')[y]

        element = self.df.iloc[index]
        complete_path = os.path.join(self.data_dir,element['img_fn'])
        img = cv2.imread(complete_path,cv2.IMREAD_UNCHANGED)

        for i in range(3):
            img[:,:,i] = img[:,:,i] 
        img = Image.fromarray(img[:,:,:3])
        class_str = element['roof_type']
        class_idx = self.classes.index(class_str)
        
        y = to_categorical(class_idx,len(self.classes))
        if self.transform is not None:
            img = self.transform(img)
        
        return img,y

    def __len__(self):
        return self.df.shape[0]

import torch
import random
import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
import numpy as np

def correct_mask(mask):
    '''
    Replace 127 with 0 for cropping purposes
    '''
    mask_arr = np.array(mask)
    mask_arr[mask_arr==127]=0
    return Image.fromarray(mask_arr)

class ClassifDataset(data.Dataset):
    def __init__(self, data_dir,mode = 'df_tagging_roof_material_type',crop_alpha=True, frac = 1, transform=None):
        
        def get_df(dirr,frac=frac):    
            df_base = pd.read_csv(dirr+'/''newroof.csv')
            df = df_base[df_base['fold_mode']==mode]
            df = df.sample(frac=frac, random_state=1,replace = False)
            return df

        super(ClassifDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.df = get_df(self.data_dir)
        self.transform = transform
        self.crop_alpha = crop_alpha
        self.classes = ['hippedRoof',
                        'gabledRoof',
                        'flatRoof',
                        'monopitchRoof',
                        'pyramidalRoof',
                        'copulaRoof',
                        'halfHippedRoof',
                        'archRoof',
                        'mansardRoof'
                                ]

    def __getitem__(self, index, color_format='RGB'):

        element = self.df.iloc[index]
        label_name =  element['img_fn']+ '.png'
        complete_path = os.path.join(self.data_dir,label_name)
        img_4c = Image.open(complete_path)
        
        if self.crop_alpha:
            alpha = correct_mask(img_4c.getchannel('A'))
            bbox = alpha.getbbox()
            img_4cc = img_4c.crop(bbox)

        else:
            img_4cc = img_4c


        img_4cc.load()
        img = Image.new(color_format, img_4cc.size, (0,0,0))
        img.paste(img_4cc,mask=img_4cc.split()[3] if self.crop_alpha else None)
        

        class_str = element['roof_type']
        class_idx = self.classes.index(class_str)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img,class_idx

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    from transforms import TransformsClassification
    import matplotlib.pyplot as plt
    trainset_classif = ClassifDataset('../new_dataset',mode='training',transform = TransformsClassification(244,'training'), frac = 1.)
    X = torch.zeros((9))
    for i in range(len(trainset_classif)):
        x,y = trainset_classif.__getitem__(i)
        #X[y] +=1/len(trainset_classif)
        plt.imshow(x.permute(1,2,0))
        #plt.title(trainset_classif.classes[y])
        plt.show()
        if i==10:
            break
    print(1/X)
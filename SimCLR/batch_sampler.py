import numpy as np
import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        print('Configuring batch sampler (this may take some time)...')
        for idx in range(0, len(dataset)):
            label = np.argmax(dataset.__getitem__(idx)[1])
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)
        print('Configuring batch sampler (this may take some time)... Done')
    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    

    def __len__(self):
        return self.balanced_max*len(self.keys)

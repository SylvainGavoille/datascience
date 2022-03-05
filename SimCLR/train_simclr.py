import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model import SimCLR
from dataset import ClassifDataset
from transforms import TransformsSimCLR
from loss import NT_Xent
from batch_sampler import BalancedBatchSampler
from lars import LARS
import matplotlib.pyplot as plt

import json

def train(device, train_loader, model, criterion, optimizer, n_epochs = 800, result_path = '../results/figs/',
             resume_epoch = 0, model_out_path = '../results/SimCLR_2048'):
    loss_per_epoch=[]
    for epoch in tqdm(range(n_epochs - resume_epoch)):
        pbar = tqdm(total = len(train_loader))
        epoch_loss = []
        for step, ((x_i, x_j), _) in enumerate(train_loader):

            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # positive pair, with encoding
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)
            loss = criterion(z_i, z_j,z_i.shape[0])

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())
            pbar.set_description(f' Epoch {epoch + 1 + resume_epoch} : Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}')
            pbar.update()
        
        loss_per_epoch.append(np.mean(epoch_loss))
        torch.save(model.state_dict(), model_out_path)
        if epoch >= 1:
            plt.figure(figsize=(10,10))
            plt.plot(list(range(1,len(loss_per_epoch) + 1)) , loss_per_epoch)
            plt.title('Train loss per epoch')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(result_path + 'SimCLR_train_loss.png')
            plt.close()
    return model






torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


if __name__ =='__main__':

    #parse config
    with open('config_simclr.json', 'r') as f:
        config = json.load(f)


    data_dir = config['data_dir']
    architecture = config['architecture']
    resume_epoch = config['resume_epoch']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    projection_dim = config['projection_dim']
    temperature = config['temperature']
    model_out_path = config['model_out_path']
    n_epochs = config['n_epochs']

    device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")
    print(device)
    print('Building model...')
    model = SimCLR(architecture, projection_dim = projection_dim).to(device)

    if resume_epoch>0:
        model.load_state_dict(torch.load(model_out_path))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print('Multiple GPUs Detected : ' , n_gpus )
        model = nn.DataParallel(model)

    print('model built')
    optimizer = LARS(model.parameters(), lr = lr, weight_decay=1e-6)
    criterion = NT_Xent(temperature, device)

    print('Configuring Dataset...')
    trainset = ClassifDataset(data_dir,transform = TransformsSimCLR(244))
    train_dl_cpc = DataLoader(trainset, batch_size=batch_size, num_workers=8, pin_memory=True)#, sampler = BalancedBatchSampler(trainset))
    print('Done ! Training will begin..')

    model = train(device=device, train_loader = train_dl_cpc, model = model, criterion = criterion, optimizer = optimizer, 
                  n_epochs = n_epochs, resume_epoch = resume_epoch, model_out_path=model_out_path)

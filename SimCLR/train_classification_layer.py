import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ClassifDataset
from transforms import TransformsClassification
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import SimCLR
import torch.optim as optim 

def train_classif(model_simclr, model_classif, train_dl, val_dl, 
                  optim, criterion, n_epochs=20, freeze_backbone = True,
                  loss_img_name = 'loss.png', acc_img_name = 'acc.png'):

    for p in model_simclr.parameters():
        p.requires_grad = not freeze_backbone

    scheduler_classif = torch.optim.lr_scheduler.MultiStepLR(optim, [35]) 
    loss_per_epoch=[]
    acc_per_epoch = []
    val_acc_per_epoch = []
    val_batches_loss = []

    for epoch in tqdm(range(n_epochs)):
        
        correct = 0
        total = 0

        pbar = tqdm(total = len(train_dl))
        epoch_loss = []

        for i, data in enumerate(train_dl):


            x = data[0].to(device)
            y = data[1].to(device)
            
            _,z = model_simclr(x)
            predictions = model_classif(z)
            y_ = torch.from_numpy(np.stack(y.cpu())).float().to(device)
            _, gt = torch.max(y_.data, 1)
            loss = criterion(nn.LogSoftmax(dim=1)(predictions),gt)
            loss.backward()
            epoch_loss.append(loss.item())

            outputs = nn.LogSoftmax(dim=1)(predictions)
            optim.step()
            optim.zero_grad()
            _, predicted = torch.max(outputs.data, 1)
            
            total += y_.size(0)
            correct += (predicted == gt).sum().item()
            pbar.set_description(' [%d, %5d] loss: %.3f - acc : %.3f ' % (epoch + 1, i + 1, loss.item(),100 * correct / total))
            pbar.update()
            
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            batch_val_loss = []
            for data in val_dl:
                images, labels = data
                z=[]
                _,z = model_simclr(images.to(device))
                logits = model_classif(z)
                outputs = nn.LogSoftmax(dim=1)(logits)
                y_ = torch.from_numpy(np.stack(labels.cpu())).float().to(device)
                _, predicted = torch.max(outputs.data, 1)
                _, gt = torch.max(y_.data, 1)
                batch_val_loss.append(criterion(outputs,gt).item())
                val_total += y_.size(0)
                val_correct += (predicted == gt).sum().item()
                
                
        mean = np.mean(batch_val_loss)
        loss_per_epoch.append(np.mean(epoch_loss))
        val_batches_loss.append(mean)
        acc_per_epoch.append(correct / total)
        val_acc_per_epoch.append(val_correct / val_total)

        print('Epoch : %d , val_loss : %f - val_acc: %d %%' % (epoch+1,mean,
            100 * val_correct / val_total))

        scheduler_classif.step()
        if epoch > 1 : 
            plt.figure(figsize=(10,10))
            plt.plot(list(range(1,epoch + 2)) , loss_per_epoch)
            plt.plot(list(range(1,epoch + 2)) , val_batches_loss)
            plt.title('Train loss per epoch')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(loss_img_name)
            plt.figure(figsize=(10,10))
            plt.plot(list(range(1,epoch + 2)) , acc_per_epoch, label = 'train')
            plt.plot(list(range(1,epoch + 2)) , val_acc_per_epoch, label = 'val')
            plt.title('acc per epoch')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.savefig(acc_img_name)
            plt.close()
    if freeze_backbone:
        return model_classif
    else:
        return model_simclr, model_classif

if __name__ == "__main__":

    #parse config
    with open('config_classif.json', 'r') as f:
        config = json.load(f)


    data_dir = config['data_dir']
    model_path = config['model_path']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    projection_dim = config['projection_dim']
    model_out_path = config['model_out_path']
    n_epochs = config['n_epochs']
    architecture = config['architecture']


    device = torch.device("cuda") #if torch.cuda.is_available() else "cpu")

    model_simclr = SimCLR(architecture).to(device)
    model_simclr = nn.DataParallel(model_simclr)
    model_simclr.load_state_dict(torch.load(model_path))



    classif_model = nn.Sequential(nn.Linear(projection_dim,9)).to(device)
    crit = nn.NLLLoss()



    trainset_classif = ClassifDataset(data_dir,mode='train',transform = TransformsClassification(244,mode='train'))
    valset = ClassifDataset(data_dir,mode='val',transform = TransformsClassification(244,mode='test'))


    train_dl_classif = DataLoader(trainset_classif, batch_size=batch_size,num_workers=8, pin_memory=True)#,sampler = BalancedBatchSampler(trainset_classif))
    val_dl = DataLoader(valset, batch_size=batch_size,num_workers=8, pin_memory=True)

    optimizer = optim.Adam(list(classif_model.parameters()), lr = lr)#+list(model.parameters())

    classif_model = train_classif(model_simclr, classif_model, train_dl_classif, val_dl,
                                 optimizer, crit, n_epochs = n_epochs, freeze_backbone = False,
                                loss_img_name = 'loss_classif.png', acc_img_name = 'acc_classif.png')

    optimizer_ft = optim.Adam(list(classif_model.parameters())+list(model.parameters()), lr = 0.1 * lr)

    _, classif_model = train_classif(model_simclr, classif_model, train_dl_classif, val_dl,
                                 optimizer, crit, n_epochs = n_epochs, freeze_backbone = True,
                                loss_img_name = 'loss_classif.png', acc_img_name = 'acc_classif.png')

    torch.save(classif_model.state_dict(), model_out_path )

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transforms import TransformsClassification
from dataset import *

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
import matplotlib.pyplot as plt
from torchvision.models import resnet34
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
import argparse
from focal import FocalLoss

class ClassifModel(pl.LightningModule):
    def __init__(self,
                 n_epochs,
                 batch_size,
                 lr,
                 crop_alpha=True,
                 criterion='nll',
                 transfer_from = 'byol',
                 data_dir = '../new_dataset',
                 init_weights_path="best_byol.ckpt",
                 freeze_encoder = True,
                 ):

        super(ClassifModel, self).__init__()

        assert transfer_from in ['byol', 'imagenet']
        assert criterion in ['nll', 'focal']

        def load_pretrained(model, pretrained_model):
            ckpt = torch.load(pretrained_model, map_location='cpu')
            model_dict = model.state_dict()
            base_fix = False
            base_learner = False
            mkey = 'model' 
            if 'model'  not in ckpt.keys():
                mkey = 'state_dict'
                
            for key in ckpt[mkey].keys():
                if key.startswith('module.base.'):
                    base_fix = True
                    break
                elif key.startswith('learner.online_encoder.net'):
                    base_learner = True
                    break
            if base_fix:
                state_dict = {k.replace("module.base.", "module."): v
                            for k, v in ckpt[mkey].items()
                            if k.startswith('module.base.')}
            elif base_learner : 
                state_dict = {k.replace("learner.online_encoder.net.", ""): v
                            for k, v in ckpt[mkey].items()
                            if k.startswith('learner.online_encoder.net')}
            else:
                state_dict = {k.replace("module.encoder.", "module."): v
                            for k, v in ckpt[mkey].items()
                            if k.startswith('module.encoder.')}
            state_dict = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.size() == model_dict[k].size()}
            model.load_state_dict(state_dict)
            return model


        def build_model():

            backbone = load_pretrained(resnet34(), init_weights_path) if (transfer_from == 'byol' and init_weights_path is not None) else resnet34(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            feature_extractor = nn.Sequential(*layers)
            linear = nn.Linear(num_filters, 9)

            if criterion =='focal':
                #Focal trick
                cst = -torch.log(torch.Tensor([(1-0.01)/0.01]))[0]
                linear.bias.data.fill_(cst) 

            classifier = nn.Sequential(nn.Flatten(),linear)

            return feature_extractor, classifier




          
            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.crit = FocalLoss(gamma=2) if criterion == 'focal' else nn.CrossEntropyLoss()
        self.backbone, self.classifier = build_model()

        self.net = nn.Sequential(self.backbone, self.classifier)

        self.freeze_encoder = freeze_encoder
        self.last_act = nn.LogSoftmax(dim=1)



        train_transform = TransformsClassification(244,'training')
        test_transform = TransformsClassification(244,'test')
        self.trainset_classif = ClassifDataset(data_dir,mode='training',transform = train_transform, frac = 1., crop_alpha=crop_alpha)
        self.valset = ClassifDataset(data_dir,mode='validation',transform = test_transform, crop_alpha=crop_alpha)
        self.testset = ClassifDataset(data_dir,mode='validation',transform = test_transform, crop_alpha=crop_alpha)


        self.train_acc = pl.metrics.Accuracy()
        self.save_hyperparameters()


    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                xx = self.backbone(x)
        else:
            xx = self.backbone(x)
        return self.classifier(xx)


    def training_step(self, batch, batch_nb):
        img, label = batch
        out = self.forward(img)
        loss = self.crit(out, label) 

        pred = self.last_act(out).detach()
        acc = self.train_acc(pred, label)
        self.log("train_overall_acc", acc , on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss',loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, label = batch
        img = img.float()
        out = self(img)
        loss_val = self.crit(out, label) 

        pred = torch.argmax(self.last_act(out)).unsqueeze(0).type_as(label)

        avg_loss = loss_val
        avg_acc  = self.train_acc(pred, label)




        self.log("val_overall_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True,logger=True)


        return {'val_loss': loss_val, 
                "val_overall_acc": avg_acc,
                'preds' : pred,
                'target': label}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        losses = torch.cat([tmp['val_loss'].reshape(1) for tmp in outputs])
        confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=9,normalize='true')

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(9), columns=range(9))
        plt.figure(figsize = (10,7))
        cm = df_cm.astype('float')
        fig_ = sns.heatmap(cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        self.log('f1_val_score', f1_score(targets.cpu(),preds.cpu(),average='macro'),on_step=False,on_epoch = True, prog_bar = True,logger=True)
        self.log('valid_loss',losses.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return super().validation_epoch_end(outputs)
    

    def test_step(self,batch,batch_idx):
        img, label = batch
        img = img.float()
        label = label
        out = self(img)
        loss_val = self.crit(out, label) 

        pred = torch.argmax(self.last_act(out)).unsqueeze(0).type_as(label)

        avg_loss = loss_val
        avg_acc  = self.train_acc(pred, label)


        self.log("test_overall_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True,logger=True)


        return {'test_loss': loss_val, 
                "test_overall_acc": avg_acc,
                'test_preds' : pred,
                'test_target': label}

    def test_epoch_end(self, outputs):
        preds = torch.cat([tmp['test_preds'] for tmp in outputs])
        targets = torch.cat([tmp['test_target'] for tmp in outputs])
        losses = torch.cat([tmp['test_loss'].reshape(1) for tmp in outputs])
        confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=9,normalize='true')

        confusion_matrix = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=9,normalize='true')

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(9), columns=range(9))
        plt.figure(figsize = (10,7))
        cm = df_cm.astype('float')
        fig_ = sns.heatmap(cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        
        self.logger.experiment.add_figure("Confusion matrix : Test Set", fig_, self.current_epoch)
        self.log('f1_test_score', f1_score(targets.cpu(),preds.cpu(),average='macro'),on_step=False,on_epoch = True, prog_bar = True,logger=True)
        self.log('test_loss',losses.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return super().test_epoch_end(outputs)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.lr, weight_decay = 1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = self.lr, epochs = self.n_epochs, steps_per_epoch = len(self.train_dataloader()))
        scheduler = {'scheduler': sch,'interval': 'step','frequency': 1}
        return [opt], [scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.trainset_classif, batch_size=self.batch_size,num_workers=12, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",'-d', help="Root folder of the dataset. where images/ and newroof.csv shall be",
                    type=str,default='../new_dataset', required=True)

    parser.add_argument("--output_dir", '-o' ,help="Path for checkpoint saving",
                    type=str,default='./results/', required=True)

    parser.add_argument("--init_weights_path",'-i',help="Relative path for initialization checkpoint",
                    type=str,default='best_byol.ckpt', required=True)

    parser.add_argument("--transfer_from",'-t',help="Whether start from ImageNET weights or BYOL weights",
                    type=str,default='byol')

    parser.add_argument("--criterion",'-c',help="Loss function : nll or focal",
                    type=str,default='nll')

    parser.add_argument("--crop_alpha",'-ca',help="Crop the image using alpha channel",
                    default=False, action='store_true')

    parser.add_argument("--no-crop_alpha",
                    dest='crop_alpha', action='store_false')

    parser.add_argument("--finetune",'-ft',help="Unfreeze encoder for training 2nd step",
                    default=False, action='store_true')

    parser.add_argument("--no-finetune",
                    dest='finetune', action='store_false')

    parser.add_argument("--n_epochs", help="Number of finetuning epochs",
                    type=int,default=20)

    parser.add_argument("--batch_size", '-b' ,help="Self-explanatory",
                    type=int,default=50)

    parser.add_argument("--lr",'-l' ,help="Base Learning Rate",
                    type=float, default=3e-4)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark=True
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(0)

    print(args)

    model = ClassifModel(
                 args.n_epochs,
                 args.batch_size,
                 args.lr,
                 transfer_from=args.transfer_from,
                 crop_alpha=args.crop_alpha,
                 data_dir=args.data_dir,
                 criterion=args.criterion,
                 init_weights_path=args.init_weights_path,
                 )
    ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='rooftypeclassif_resnet34-tf-{epoch:02d}',
        monitor='valid_loss',
    )
    lr_logger = LearningRateMonitor(logging_interval='step')



    trainer_tl = pl.Trainer(
        gpus = 1,
        max_epochs = args.n_epochs,
        accumulate_grad_batches = 1,
        sync_batchnorm = False,
        callbacks = [ckpt_callback, lr_logger],
        accelerator = None,
        log_every_n_steps=10,
        precision=16
    )
    trainer_tl.fit(model)
    trainer_tl.test()
    
    if args.finetune:
        model.freeze_encoder = False
        trainer_ft = pl.Trainer(
            gpus = 1,
            max_epochs = args.n_epochs,
            accumulate_grad_batches = 1,
            sync_batchnorm = False,
            callbacks = [ckpt_callback, lr_logger],
            accelerator = None,
            log_every_n_steps=10,
            precision=16
        )
        trainer_ft.fit(model)
        trainer_ft.test()
    
    

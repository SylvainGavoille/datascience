from main import ClassifModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

import argparse




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",'-d' ,help="Root folder of the dataset. where images/ and newroof.csv shall be",
                    type=str,default='../new_dataset')

    parser.add_argument("--ckpt_path", '-c',help="Path for model checkpoint",
                    type=str,default='./results/rooftypeclassif_resnet34-tf-epoch=18.ckpt')
    parser.add_argument("--n_epochs", help="Number of finetuning epochs",
                    type=int,default=20)
    parser.add_argument("--batch_size", '-b' ,help="Number of finetuning epochs",
                    type=int,default=100)
    parser.add_argument("--lr",'-l' ,help="Base Learning Rate",
                    type=float, default=3e-4)
    parser.add_argument("--output_dir", '-o' ,help="Path for checkpoint saving",
                    type=str,default='./results/', required=False)


    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr



    model = ClassifModel(
                n_epochs,
                batch_size,
                lr,
                data_dir = args.data_dir, 
                freeze_encoder=False
                )
    model_test = model.load_from_checkpoint(args.ckpt_path) 
    model_test.freeze_encoder = False # Just for being doubleplus sure 

    ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='rooftypeclassif_resnet34-ft-{epoch:02d}',
        monitor='valid_loss',
    )
    lr_logger = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus = 1,
        max_epochs = n_epochs,
        accumulate_grad_batches = 1,
        sync_batchnorm = False,
        callbacks = [ckpt_callback, lr_logger],
        accelerator = None,
        log_every_n_steps=10,
        precision=16
    )
    trainer.fit(model_test)
    trainer.test()

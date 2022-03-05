from main import ClassifModel

import pytorch_lightning as pl

import argparse




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",'-d' ,help="Root folder of the dataset. where images/ and newroof.csv shall be",
                    type=str,default='../new_dataset')

    parser.add_argument("--ckpt_path", '-c',help="Path for model checkpoint",
                    type=str,default='../results_byol_finetuneig/rooftypeclassif_resnet34-epoch=13.ckpt')

    parser.add_argument("--init_weights_path",'-i',help="Relative path for initialization checkpoint",
                    type=str,default=None)

    args = parser.parse_args()

    # Dummy vars for model Definition
    n_epochs = 10
    batch_size = 10
    lr = 10



    model = ClassifModel(
                n_epochs,
                batch_size,
                lr,
                data_dir = args.data_dir, #Change to actual data dir
                init_weights_path=args.init_weights_path,
                )

    trainer = pl.Trainer(gpus=1)
    model_test = model.load_from_checkpoint(args.ckpt_path, data_dir=args.data_dir, init_weights_path= args.init_weights_path) #Change to sent checkpoint path
    trainer.test(model_test)

    '''
    Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████▊| 3858/3864 [00:28<00:00, 137.90it/s]--------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'final_f1_test_score': 0.7509102573829628,
    'test_loss': tensor(0.2889, device='cuda:0'),
    'test_overall_acc': tensor(0.9128, device='cuda:0')}
    --------------------------------------------------------------------------------
    '''

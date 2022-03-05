Code for Fine Tuning BYOL-pretrained resnet34 for rooftype Classification

To run this code (minimal args):

1. Download BYOL weights : https://drive.google.com/file/d/10aG-LG7liIdLAxsFmzHRP-w5yDAqQTQG/view?usp=sharing
2. Move it to byol_finetuning folder
3. Download newroof.csv https://drive.google.com/file/d/1g0yaLVVq9RgXIbFsWOLwiOhXGdcWhWz2/view?usp=sharing
3. Put newroof.csv into the same parent directory as images/
4. run : 

```
pip install -r requirements.txt
```
Then : 

```
python main.py -d [DATA_PATH] -o [RESULTS_PATH] -i best_byol.ckpt --transfer_from byol -c nll --no-crop_alpha --no-finetuning --batch_size 50
```
or (for imagenet)
```
python main.py -d [DATA_PATH] -o [RESULTS_PATH] -i best_byol.ckpt --transfer_from imagenet -c nll --no-crop_alpha --finetuning
```


```
python evaluate.py -d [DATA_PATH] -c [ROOT_CHECKPOINT_PATH]
```

Median score : 91% accuracy , 0.75 macro-averaged f1 score

For more help with execution : 


```
python main.py -h
python evaluate.py -h
```



TODO : 
1. Add requirements.txt [DONE]
2. Use of argparser for smoother execution [DONE]


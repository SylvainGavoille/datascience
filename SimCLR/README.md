Project for training SimCLR architecture on Roof Type Classification Dataset.

Inspired from : https://github.com/sthalles/SimCLR

To run the project : 

1- Clone the repository and navigate to SimCLR directory

    git clone https://github.com/cmla/namr-segmentation.git
    cd namr-segmentation/SimCLR

2- Install requirements (you may want to create a virtual env for the project before) :
    
    python3 -m pip install -r req.txt
    
3- Change hyperparameters and variables in ``config_simclr.json ``

4- Run ``train_simclr.py ``

    python3 train_simclr.py
    

After training using SimCLR, you can train a classification layer on top of learned features : 

1- Change config in ``config_classif.json ``

2- Run ``train_classification_layer.py ``

    python3 train_classification_layer.py 

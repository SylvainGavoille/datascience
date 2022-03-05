import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

def plot_confusion_matrix(test_dl, normalize = True):
    nb_classes = 9

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            z=[]
            _,z = model(images.to(device))
            logits = classif_model(z)
            outputs = nn.LogSoftmax(dim=1)(logits)
            y_ = torch.from_numpy(np.stack(labels.cpu())).float().to(device)
            _, preds = torch.max(outputs.data, 1)
            _, gt = torch.max(y_.data, 1)
            for t, p in zip(gt.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(1).view((9,1))

    plt.figure(figsize=(10,10))
    plt.title('Confusion Matrix : Test Set')
    sns.heatmap(xticklabels = classes,yticklabels=classes,data=confusion_matrix,annot=True, cmap="YlGnBu",cbar=False)
    return(confusion_matrix)

def get_test_acc(model_simclr, model_classif, test_dl);
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            z=[]
            _,z = model(images.to(device))
            logits = model_classif(z)
            outputs = nn.LogSoftmax(dim=1)(logits)
            y_ = torch.from_numpy(np.stack(labels.cpu())).float().to(device)
            _, predicted = torch.max(outputs.data, 1)
            _, gt = torch.max(y_.data, 1)
            test_total += y_.size(0)
            test_correct += (predicted == gt).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * test_correct / test_total))

testset = ClassifDataset(data_dir,mode='test',transform = TransformsClassification(244,mode='test'))
test_dl = DataLoader(testset, batch_size=10,shuffle=False)


model_simclr = SimCLR('resnet18').to(device)
model_simclr.load_state_dict(torch.load('./results/SimCLR_new_dataset_balanced_sampler'))

classif_model = nn.Sequential(nn.Linear(64,9)).to(device)
classif_model.load_state_dict(torch.load('./results/SimCLR_classif_layer_new_dataset_balanced_sampler'))

get_test_acc(model_simclr, classif_model, test_dl)
import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, resnet, projection_dim = 128, normalize=True):
        super(SimCLR, self).__init__()


        self.encoder = self.get_resnet(resnet)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer
        self.projection_dim = projection_dim
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.projection_dim, bias=False),
        )
        self.normalize = normalize

    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
            "resnet34": torchvision.models.resnet34(),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]


    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z

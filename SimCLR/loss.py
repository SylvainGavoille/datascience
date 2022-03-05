import torch
import torch.nn as nn

#TODO add ref

class NT_Xent(nn.Module):
    def __init__(self, temperature, device):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def mask_correlated_samples(self, size):
        mask = torch.ones((size * 2, size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        return mask


    def forward(self, z_i, z_j,size):
        
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """


        mask = self.mask_correlated_samples(size)
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, size)
        sim_j_i = torch.diag(sim, -size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            size * 2, 1
        )
        negative_samples = sim[mask].reshape(size * 2, -1)

        labels = torch.zeros(size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * size
        return loss
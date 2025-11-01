import torch
import torch.nn as nn



# distribution loss
class dist_loss(nn.Module):
    def __init__(self):
        super(dist_loss, self).__init__()
    
    def forward(self, outputs, y):
        mu, sigma = outputs
        l_mean = (mu - y) ** 2 # (B, output_steps, 2) for highD
        l_std = (torch.abs(mu - y) - sigma)**2
        loss = l_mean + l_std
        
        return torch.mean(loss)
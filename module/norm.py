import torch
import torch.nn as nn
import torch.nn.functional as F


class InstNorm(nn.Module):
    def __init__(self):
        super(InstNorm, self).__init__()

    def norm(self, x):
        self.means = x.mean(1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / self.stdev
        return x
    
    def denorm(self, x):
        pred_len = x.shape[1]
        x = x * (self.stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
        x = x + (self.means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1))
        return x
import torch
from torch.functional import mse_loss


def get_hidden_distill_loss(T_proj, S_proj):
    hidden_loss = 0
    for i in range(T_proj.size()):
        hidden_loss += mse_loss(S_proj[i], T_proj[i])

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-15


class Contrast(nn.Module):
    def __init__(self, hidden_dim, project_dim, act, tau):
        super(Contrast, self).__init__()
        self.tau = tau
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            act,
            nn.Linear(project_dim, project_dim)
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            act,
            nn.Linear(project_dim, project_dim)
        )
        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
        for model in self.proj_2:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_1, z_2, pos):
        z_proj_1 = self.proj_1(z_1)
        z_proj_2 = self.proj_2(z_2)
        matrix_1 = self.sim(z_proj_1, z_proj_2)
        matrix_2 = matrix_1.t()


        matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EPS)
        lori_1 = -torch.log(matrix_1.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EPS)
        lori_2 = -torch.log(matrix_2.mul(pos.to_dense()).sum(dim=-1)).mean()


        return (lori_1 + lori_2) / 2


def sce_loss(x, y, beta=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)

    loss = loss.mean()
    # to balance the loss
    return 10 * loss



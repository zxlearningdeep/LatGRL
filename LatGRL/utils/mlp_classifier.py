import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(MLP, self).__init__()
        hidden_dim_1 = 512
        self.fc = nn.Sequential(
                                nn.BatchNorm1d(ft_in),
                                nn.Linear(ft_in, hidden_dim_1),
                                nn.ELU(),
                                nn.BatchNorm1d(hidden_dim_1),
                                nn.Linear(hidden_dim_1, nb_classes))

        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


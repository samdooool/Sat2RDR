import torch
from torch import nn

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        diff_log = torch.log(target+1) - torch.log(pred+1)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class GANLoss(object):
    def __init__(self,
            n_D = 1):
        self.device = torch.device('cuda:0')
        self.dtype = torch.float32

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.lambda_FM = 10
        self.n_D = n_D

    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * (1.0 / self.n_D) * self.lambda_FM

        return loss_D, loss_G, target, fake

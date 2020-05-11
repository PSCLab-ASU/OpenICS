import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 1, 3, 3))
        )
        self.conv2_forward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3))
        )
        self.conv1_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3))
        )
        self.conv2_backward = nn.Parameter(
            init.xavier_normal_(torch.Tensor(1, 32, 3, 3))
        )

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(
            torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr)
        )

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, 1089)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


class ISTANetModel(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetModel, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):
        # print("PHIX IN MODEL", Phix.size(), "PHI SIZE IN MODEL", Phi.size())

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        # print("PHIX BEFORE BREAKING", Phix.size(), "PHI BEFORE BREAKING", Phi.size())

        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []  # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]

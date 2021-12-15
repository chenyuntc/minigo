# code based on https://github.com/CGLemon/pyDLGO/blob/master/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_CHANNELS, BLOCK_SIZE, FILTER_SIZE, USE_GPU


class FullyConnect(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        relu=True,
    ):
        super().__init__()
        self.relu = relu
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True) if self.relu else x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        relu=True,
    ):
        super().__init__()
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) if self.relu else x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
        )
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            relu=False,
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(
        self,
        board_size,
        input_channels=INPUT_CHANNELS,
        block_size=BLOCK_SIZE,
        filter_size=FILTER_SIZE,
        use_gpu=USE_GPU,
    ):
        super().__init__()

        self.block_size = block_size
        self.residual_channels = filter_size
        self.policy_channels = 8
        self.value_channels = 4
        self.value_layers = 256
        self.board_size = board_size
        self.spatial_size = self.board_size ** 2
        self.input_channels = input_channels
        self.use_gpu = True if torch.cuda.is_available() and use_gpu else False
        self.gpu_device = torch.device("cpu")

        self.build_network()
        if self.use_gpu:
            self.gpu_device = torch.device("cuda")
            self.cuda()

    def build_network(self):
        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            relu=True,
        )

        # residual tower
        nn_stack = [ResBlock(self.residual_channels) for _ in range(self.block_size)]
        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_channels,
            kernel_size=1,
            relu=True,
        )
        self.policy_fc = FullyConnect(
            in_size=self.policy_channels * self.spatial_size,
            out_size=self.spatial_size + 1,
            relu=False,
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            relu=True,
        )

        self.value_fc = FullyConnect(
            in_size=self.value_channels * self.spatial_size,
            out_size=self.value_layers,
            relu=True,
        )
        self.winrate_fc = FullyConnect(
            in_size=self.value_layers,
            out_size=1,
            relu=False,
        )

    def forward(self, planes):
        x = self.input_conv(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1, end_dim=3))

        # value head
        val = self.value_conv(x)
        val = self.value_fc(torch.flatten(val, start_dim=1, end_dim=3))
        val = self.winrate_fc(val)

        return pol, torch.tanh(val)

    def get_outputs(self, planes):
        m = nn.Softmax(dim=1)
        x = torch.unsqueeze(torch.tensor(planes, dtype=torch.float32), dim=0)
        if self.use_gpu:
            x = x.cuda()
        p, v = self.forward(x)
        return m(p).item(), v.item()

    def trainable(self, t=True):
        torch.set_grad_enabled(t)
        if t == True:
            self.train()
        else:
            self.eval()

    def save_ckpt(self, filename):
        torch.save(self.state_dict(), filename)

    def load_ckpt(self, filename):
        self.load_state_dict(torch.load(filename, map_location="cpu"))

from fixmatch.models.wideresnet import *


class Confucius(nn.Module):
    def __init__(self, output_dim, expose_dim, hidden):
        super(Confucius, self).__init__()
        self.output_fc = nn.Linear(output_dim, hidden)
        self.fc_expose = nn.Linear(expose_dim, hidden)
        self.fc_final = nn.Linear(hidden, 1)

    def forward(self, output, expose):
        out1 = self.output_fc(output)
        out2 = self.fc_expose(expose)
        out = self.fc_final(out1 + out2)
        return out


class WideResConfucius(nn.Module):
    def __init__(self, main_wide_res_net):
        super(WideResConfucius, self).__init__()
        # parasite the main wideresnet
        self.main_wrn = main_wide_res_net
        depth = main_wide_res_net.depth
        widen_factor = main_wide_res_net.widen_factor
        drop_rate = main_wide_res_net.drop_rate
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.channels = channels
        self.n = (depth - 4) / 6
        block = BasicBlock

        self.block23 = NetworkBlock(
            self.n, channels[1], channels[3], block, 2, drop_rate, activate_before_residual=True)
        self.output_fc_1 = nn.Linear(channels[3], channels[3])
        self.output_fc_2 = nn.Linear(channels[3], 1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, output, e1, e3):
        e33 = self.relu(self.block23(e1))
        e33 = F.adaptive_avg_pool2d(e33, 1)
        e33 = e33.squeeze(-1).squeeze(-1)
        comb = e33 + e3
        out = self.relu(self.output_fc_1(comb))
        out = self.output_fc_2(out)
        return out

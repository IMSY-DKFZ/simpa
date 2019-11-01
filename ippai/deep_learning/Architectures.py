import torch.nn as nn
from torch import cat as concatenate
import torch.nn.init as init
from collections import OrderedDict


class NamedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def name(self):
        return "NamedNeuralNetwork"


class AutoEncoder(NamedNeuralNetwork):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        internal_channels = 32

        self.contract_1 = self.contract(in_channels, internal_channels*4)
        self.contract_2 = self.contract(internal_channels*4, internal_channels * 8)
        self.contract_3 = self.contract(internal_channels*8, internal_channels * 16)
        self.contract_4 = self.contract(internal_channels*16, internal_channels * 32)
        self.expand_4 = self.expand(internal_channels * 32, internal_channels * 16)
        self.expand_3 = self.expand(internal_channels * 16, internal_channels * 8)
        self.expand_2 = self.expand(internal_channels * 8, internal_channels * 4)
        self.expand_1 = self.expand(internal_channels * 4, in_channels)

        self.final = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=2,
                          padding=int(kernel_size/2)),
                nn.LeakyReLU(inplace=True))

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2,
                               stride=2),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.contract_1(x)
        x = self.contract_2(x)
        x = self.contract_3(x)
        x = self.contract_4(x)
        x = self.expand_4(x)
        x = self.expand_3(x)
        x = self.expand_2(x)
        x = self.expand_1(x)
        x = self.final(x)
        return x

    def name(self):
        return "AutoEncoder_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)


class FullyCNN(NamedNeuralNetwork):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.contract_1 = self.contract(in_channels, in_channels*4)
        self.contract_2 = self.contract(in_channels*4, in_channels * 8)
        self.contract_3 = self.contract(in_channels*8, in_channels * 16)
        self.contract_4 = self.contract(in_channels*16, in_channels * 8)
        self.contract_5 = self.contract(in_channels*8, in_channels * 4)
        self.contract_6 = self.contract(in_channels*4, out_channels)
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size = 3):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
                nn.LeakyReLU())

    def forward(self, x):
        x = self.contract_1(x)
        x = self.contract_2(x)
        x = self.contract_3(x)
        x = self.contract_4(x)
        x = self.contract_5(x)
        x = self.contract_6(x)
        x = self.final(x)
        return x

    def name(self):
        return "FullyCNN_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)


class SRCNN(NamedNeuralNetwork):
    def __init__(self, depth=2, channel=1):
        super(SRCNN, self).__init__()

        self.patch_layer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1),
                                         nn.ReLU())
        self.mapping_layer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
                                         nn.ReLU())
        self.reconstruction_layer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1)

    def forward(self, x):
        x = self.patch_layer(x)
        x = self.mapping_layer(x)
        x = self.reconstruction_layer(x)
        return x

    def name(self):
        return "SRCNN"


class ESPCN(NamedNeuralNetwork):
    def __init__(self, upscale_factor=2):
        super(NamedNeuralNetwork, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.pad = nn.ReplicationPad2d((0, 1, 0, 1))
        self.out_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = self.pad(x)
        x = self.out_layer(x)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
        init.constant_(self.out_layer.weight, 1)

    def name(self):
        return "ESPCN"


class UNet(NamedNeuralNetwork):
    def __init__(self, in_channels=1, out_channels=1, internal_channels=16, dropout=0,
                 skip_kernel_size=2, skip_stride=1, skip_padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 3
        self.contract_1_1 = self.contract(in_channels, internal_channels, kernel_size, dropout=dropout)
        self.contract_1_2 = self.contract(internal_channels, internal_channels, kernel_size, dropout=dropout)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_1 = self.skip(channels=internal_channels, skip_kernel_size=2,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_2_1 = self.contract(internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.contract_2_2 = self.contract(2*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_2 = self.skip(channels=2*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_3_1 = self.contract(2*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.contract_3_2 = self.contract(4*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_3 = self.skip(channels=4*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_4_1 = self.contract(4*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.contract_4_2 = self.contract(8*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_4 = self.skip(channels=8*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.center = nn.Sequential(
            nn.Conv2d(8*internal_channels, 16*internal_channels, 3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
            self.skip(16*internal_channels, skip_kernel_size, skip_stride, skip_padding, dropout),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16*internal_channels, 8*internal_channels, kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

        self.expand_4_1 = self.expand(16*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.expand_4_2 = self.expand(8*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.upscale_4 = nn.ConvTranspose2d(8*internal_channels, 4*internal_channels, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(8*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.expand_3_2 = self.expand(4*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.upscale_3 = nn.ConvTranspose2d(4*internal_channels, 2*internal_channels, kernel_size=2, stride=2)

        self.expand_2_1 = self.expand(4*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.expand_2_2 = self.expand(2*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.upscale_2 = nn.ConvTranspose2d(2*internal_channels, internal_channels, kernel_size=2, stride=2)

        self.expand_1_1 = self.expand(2*internal_channels, internal_channels, kernel_size, dropout=dropout)
        self.expand_1_2 = self.expand(internal_channels, internal_channels, kernel_size, dropout=dropout)

        self.final = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1),
            nn.LeakyReLU()
        )

    @staticmethod
    def contract(in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
                nn.Dropout2d(p=dropout),
                nn.LeakyReLU())

    @staticmethod
    def expand(in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

    @staticmethod
    def skip(channels, skip_kernel_size, skip_stride, skip_padding, dropout):
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels,
                      kernel_size=skip_kernel_size,
                      stride=2)
        )

    def forward(self, x):
        contract_1_2 = self.contract_1_2(self.contract_1_1(x))
        pool_1 = self.pool_1(contract_1_2)
        skip_1 = self.skip_1(contract_1_2)

        contract_2_2 = self.contract_2_2(self.contract_2_1(pool_1))
        pool_2 = self.pool_2(contract_2_2)
        skip_2 = self.skip_2(contract_2_2)

        contract_3_2 = self.contract_3_2(self.contract_3_1(pool_2))
        pool_3 = self.pool_3(contract_3_2)
        skip_3 = self.skip_3(contract_3_2)

        contract_4_2 = self.contract_4_2(self.contract_4_1(pool_3))
        pool_4 = self.pool_4(contract_4_2)
        skip_4 = self.skip_4(contract_4_2)

        center = self.center(pool_4)

        concatenated_skip_4 = concatenate([center, skip_4], 1)
        expand_4_1 = self.expand_4_1(concatenated_skip_4)
        expand_4_2 = self.expand_4_2(expand_4_1)
        upscale_4 = self.upscale_4(expand_4_2)

        concatenated_skip_3 = concatenate([upscale_4, skip_3], 1)
        expand_3_1 = self.expand_3_1(concatenated_skip_3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        upscale3 = self.upscale_3(expand_3_2)

        concatenated_skip_2 = concatenate([upscale3, skip_2], 1)
        expand_2_1 = self.expand_2_1(concatenated_skip_2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        upscale2 = self.upscale_2(expand_2_2)

        concatenated_skip_1 = concatenate([upscale2, skip_1], 1)
        expand_1_1 = self.expand_1_1(concatenated_skip_1)
        expand_1_2 = self.expand_1_2(expand_1_1)

        output = self.final(expand_1_2)

        return output

    def name(self):
        return "UNet_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)


class AsymUNet(NamedNeuralNetwork):
    def __init__(self, in_channels=1, out_channels=1, internal_channels=16, dropout=0.0, output_asym_factor=20):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 3  # 3x3 conv
        self.contract_1_1 = self.contract(in_channels, internal_channels, kernel_size, dropout)
        self.contract_1_2 = self.contract(internal_channels, internal_channels, kernel_size, dropout)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_1 = self.skip(internal_channels, dropout, output_asym_factor)

        self.contract_2_1 = self.contract(internal_channels, 2*internal_channels, kernel_size, dropout)
        self.contract_2_2 = self.contract(2*internal_channels, 2*internal_channels, kernel_size, dropout)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_2 = self.skip(2*internal_channels, dropout, output_asym_factor)

        self.contract_3_1 = self.contract(2*internal_channels, 4*internal_channels, kernel_size, dropout)
        self.contract_3_2 = self.contract(4*internal_channels, 4*internal_channels, kernel_size, dropout)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_3 = self.skip(4*internal_channels, dropout, output_asym_factor)

        self.contract_4_1 = self.contract(4*internal_channels, 8*internal_channels, kernel_size, dropout)
        self.contract_4_2 = self.contract(8*internal_channels, 8*internal_channels, kernel_size, dropout)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_4 = self.skip(8*internal_channels, dropout, output_asym_factor)

        self.center = nn.Sequential(
            nn.Conv2d(8*internal_channels, 16*internal_channels, 3, padding=1),
            nn.ReLU(),
            self.skip(16*internal_channels, dropout, output_asym_factor),
            nn.ReLU(),
            nn.ConvTranspose2d(16*internal_channels, 8*internal_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.expand_4_1 = self.expand(16*internal_channels, 8*internal_channels, kernel_size, dropout)
        self.expand_4_2 = self.expand(8*internal_channels, 8*internal_channels, kernel_size, dropout)
        self.upscale_4 = nn.ConvTranspose2d(8*internal_channels, 4*internal_channels, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(8*internal_channels, 4*internal_channels, kernel_size, dropout)
        self.expand_3_2 = self.expand(4*internal_channels, 4*internal_channels, kernel_size, dropout)
        self.upscale_3 = nn.ConvTranspose2d(4*internal_channels, 2*internal_channels, kernel_size=2, stride=2)

        self.expand_2_1 = self.expand(4*internal_channels, 2*internal_channels, kernel_size, dropout)
        self.expand_2_2 = self.expand(2*internal_channels, 2*internal_channels, kernel_size, dropout)
        self.upscale_2 = nn.ConvTranspose2d(2*internal_channels, internal_channels, kernel_size=2, stride=2)

        self.expand_1_1 = self.expand(2*internal_channels, internal_channels, kernel_size, dropout)
        self.expand_1_2 = self.expand(internal_channels, internal_channels, kernel_size, dropout)
        self.final = nn.Conv2d(internal_channels, out_channels, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size, dropout=0.0):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
                nn.Dropout2d(p=dropout),
                nn.LeakyReLU())

    @staticmethod
    def expand(in_channels, out_channels, kernel_size, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

    @staticmethod
    def skip(channels, dropout=0.0, output_asym_factor=20):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(output_asym_factor, 3), stride=(output_asym_factor, 1),
                      padding=(round(output_asym_factor/2) - 1, 1)),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        contract_1_2 = self.contract_1_2(self.contract_1_1(x))
        pool_1 = self.pool_1(contract_1_2)
        skip_1 = self.skip_1(contract_1_2)

        contract_2_2 = self.contract_2_2(self.contract_2_1(pool_1))
        pool_2 = self.pool_2(contract_2_2)
        skip_2 = self.skip_2(contract_2_2)

        contract_3_2 = self.contract_3_2(self.contract_3_1(pool_2))
        pool_3 = self.pool_3(contract_3_2)
        skip_3 = self.skip_3(contract_3_2)

        contract_4_2 = self.contract_4_2(self.contract_4_1(pool_3))
        pool_4 = self.pool_4(contract_4_2)
        skip_4 = self.skip_4(contract_4_2)

        center = self.center(pool_4)

        # print(contract_4_2.size())
        # print("pool_1", pool_1.size())
        # print("pool_2", pool_2.size())
        # print("pool_3", pool_3.size())
        # print("pool_4", pool_4.size())
        # print("center", center.size())
        # print("skip_4", skip_4.size())
        # print("skip_3", skip_3.size())
        # print("skip_2", skip_2.size())
        # print("skip_1", skip_1.size())

        concatenated_skip_4 = concatenate([center, skip_4], 1)
        expand_4_1 = self.expand_4_1(concatenated_skip_4)
        expand_4_2 = self.expand_4_2(expand_4_1)
        upscale_4 = self.upscale_4(expand_4_2)

        concatenated_skip_3 = concatenate([upscale_4, skip_3], 1)
        expand_3_1 = self.expand_3_1(concatenated_skip_3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        upscale3 = self.upscale_3(expand_3_2)

        concatenated_skip_2 = concatenate([upscale3, skip_2], 1)
        expand_2_1 = self.expand_2_1(concatenated_skip_2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        upscale2 = self.upscale_2(expand_2_2)

        concatenated_skip_1 = concatenate([upscale2, skip_1], 1)
        expand_1_1 = self.expand_1_1(concatenated_skip_1)
        expand_1_2 = self.expand_1_2(expand_1_1)

        output = self.final(expand_1_2)

        return output

    def name(self):
        return "AsymUNet_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)


class AsymAutoEncoder(NamedNeuralNetwork):
    def __init__(self, in_xdim, in_ydim, out_xdim, out_ydim,
                 in_channels=1, out_channels=1, internal_channels=16, dropout_in_center=0.0):
        super().__init__()

        if not (self.check_divisible_by_2_to_the_power_5(in_xdim) and
                self.check_divisible_by_2_to_the_power_5(in_ydim) and
                self.check_divisible_by_2_to_the_power_5(out_xdim) and
                self.check_divisible_by_2_to_the_power_5(out_ydim)):
            raise AssertionError("Input and output sizes need to be divisible by 16!")

        self.in_xdim = in_xdim
        self.in_ydim = in_ydim
        self.out_xdim = out_xdim
        self.out_ydim = out_ydim
        self.dropout = dropout_in_center

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.internal_channels = internal_channels

        kernel_size = 3
        self.contract_1_1 = self.convolve(in_channels, internal_channels, kernel_size)
        self.contract_1_2 = self.convolve(internal_channels, internal_channels, kernel_size)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_2_1 = self.convolve(internal_channels, 2 * internal_channels, kernel_size)
        self.contract_2_2 = self.convolve(2 * internal_channels, 2 * internal_channels, kernel_size)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_3_1 = self.convolve(2 * internal_channels, 4 * internal_channels, kernel_size)
        self.contract_3_2 = self.convolve(4 * internal_channels, 4 * internal_channels, kernel_size)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_4_1 = self.convolve(4 * internal_channels, 8 * internal_channels, kernel_size)
        self.contract_4_2 = self.convolve(8 * internal_channels, 8 * internal_channels, kernel_size)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center_0 = self.convolve(8 * internal_channels, 8 * internal_channels, kernel_size)
        self.center_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_2 = self.skip(8*internal_channels, in_xdim*in_ydim / 256, out_xdim*out_ydim / 256, self.dropout)
        self.center_3 = nn.ConvTranspose2d(8*internal_channels,8*internal_channels, kernel_size=2, stride=2)
        self.center_4 = self.convolve(8 * internal_channels, 8 * internal_channels, kernel_size)

        self.upscale_4 = nn.ConvTranspose2d(8 * internal_channels, 8 * internal_channels, kernel_size=2, stride=2)
        self.expand_4_1 = self.expand(8*internal_channels, 8*internal_channels, kernel_size)
        self.expand_4_2 = self.expand(8*internal_channels, 8*internal_channels, kernel_size)

        self.upscale_3 = nn.ConvTranspose2d(8*internal_channels, 4*internal_channels, kernel_size=2, stride=2)
        self.expand_3_1 = self.expand(4*internal_channels, 4*internal_channels, kernel_size)
        self.expand_3_2 = self.expand(4*internal_channels, 4*internal_channels, kernel_size)

        self.upscale_2 = nn.ConvTranspose2d(4*internal_channels, 2*internal_channels, kernel_size=2, stride=2)
        self.expand_2_1 = self.expand(2*internal_channels, 2*internal_channels, kernel_size)
        self.expand_2_2 = self.expand(2*internal_channels, 2*internal_channels, kernel_size)

        self.upscale_1 = nn.ConvTranspose2d(2*internal_channels, internal_channels, kernel_size=2, stride=2)
        self.expand_1_1 = self.expand(internal_channels, internal_channels, kernel_size)
        self.expand_1_2 = self.expand(internal_channels, internal_channels, kernel_size)
        self.final = nn.Conv2d(internal_channels, out_channels, kernel_size=1)

    @staticmethod
    def check_divisible_by_2_to_the_power_5(number):
        return number % 32 == 0

    @staticmethod
    def convolve(in_channels, out_channels, kernel_size):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
                nn.LeakyReLU())

    @staticmethod
    def expand(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
            nn.LeakyReLU(),
        )

    @staticmethod
    def skip(channels, in_size, out_size, dropout):
        return nn.Sequential(
            nn.Linear(int(in_size*channels / 4), int(out_size*channels / 4)),
            nn.Dropout(p=dropout),
            nn.LeakyReLU()
        )

    def forward(self, x):

        contract_1_2 = self.contract_1_2(self.contract_1_1(x))
        pool_1 = self.pool_1(contract_1_2)

        contract_2_2 = self.contract_2_2(self.contract_2_1(pool_1))
        pool_2 = self.pool_2(contract_2_2)

        contract_3_2 = self.contract_3_2(self.contract_3_1(pool_2))
        pool_3 = self.pool_3(contract_3_2)

        contract_4_2 = self.contract_4_2(self.contract_4_1(pool_3))
        pool_4 = self.pool_4(contract_4_2)

        center_0 = self.center_0(pool_4)
        center_1 = self.center_1(center_0)
        center_1 = center_1.view(-1, self.internal_channels * 8 * self.in_xdim*self.in_ydim / 1024)
        center_2 = self.center_2(center_1)
        center_2 = center_2.view(-1, self.internal_channels * 8, self.out_xdim / 32, self.out_ydim / 32)
        center_3 = self.center_3(center_2)

        upscale_4 = self.upscale_4(center_3)
        expand_4_2 = self.expand_4_2(self.expand_4_1(upscale_4))

        upscale3 = self.upscale_3(expand_4_2)
        expand_3_2 = self.expand_3_2(self.expand_3_1(upscale3))

        upscale2 = self.upscale_2(expand_3_2)
        expand_2_2 = self.expand_2_2(self.expand_2_1(upscale2))

        upscale1 = self.upscale_1(expand_2_2)
        expand_1_2 = self.expand_1_2(self.expand_1_1(upscale1))

        output = self.final(expand_1_2)

        return output

    def name(self):
        return "AsymAutoEncoder_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)


class FeedForwardRegressor2(NamedNeuralNetwork):
    def __init__(self, input_size, output_size=1, hidden_layer_relative_size=4):
        super(FeedForwardRegressor2, self).__init__()

        hidden_size = int(hidden_layer_relative_size * input_size)

        self.upscale_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU())

        self.hidden_layer_1 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_2 = self.hidden_layer(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def hidden_layer(in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.upscale_layer(out)

        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)

        out = self.output_layer(out)
        return out

    def name(self):
        return "FeedForwardRegressor2"


class FeedForwardRegressor4(NamedNeuralNetwork):
    def __init__(self, input_size, output_size=1, hidden_layer_relative_size=4):
        super(FeedForwardRegressor4, self).__init__()

        hidden_size = int(hidden_layer_relative_size * input_size)

        self.upscale_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU())

        self.hidden_layer_1 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_2 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_3 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_4 = self.hidden_layer(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def hidden_layer(in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.upscale_layer(out)

        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)
        out = self.hidden_layer_3(out)
        out = self.hidden_layer_4(out)

        out = self.output_layer(out)
        return out

    def name(self):
        return "FeedForwardRegressor4"


class FeedForwardRegressor8(NamedNeuralNetwork):
    def __init__(self, input_size, output_size=1, hidden_layer_relative_size=4):
        super(FeedForwardRegressor8, self).__init__()

        hidden_size = int(hidden_layer_relative_size * input_size)

        self.upscale_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU())

        self.hidden_layer_1 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_2 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_3 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_4 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_5 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_6 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_7 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_8 = self.hidden_layer(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def hidden_layer(in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.upscale_layer(out)

        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)
        out = self.hidden_layer_3(out)
        out = self.hidden_layer_4(out)
        out = self.hidden_layer_5(out)
        out = self.hidden_layer_6(out)
        out = self.hidden_layer_7(out)
        out = self.hidden_layer_8(out)

        out = self.output_layer(out)
        return out

    def name(self):
        return "FeedForwardRegressor8"


class FeedForwardRegressor12(NamedNeuralNetwork):
    def __init__(self, input_size, output_size=1, hidden_layer_relative_size=4):
        super(FeedForwardRegressor12, self).__init__()

        hidden_size = int(hidden_layer_relative_size * input_size)

        self.upscale_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU())

        self.hidden_layer_1 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_2 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_3 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_4 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_5 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_6 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_7 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_8 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_9 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_10 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_11 = self.hidden_layer(hidden_size, hidden_size)
        self.hidden_layer_12 = self.hidden_layer(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def hidden_layer(in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.upscale_layer(out)

        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)
        out = self.hidden_layer_3(out)
        out = self.hidden_layer_4(out)
        out = self.hidden_layer_5(out)
        out = self.hidden_layer_6(out)
        out = self.hidden_layer_7(out)
        out = self.hidden_layer_8(out)
        out = self.hidden_layer_9(out)
        out = self.hidden_layer_10(out)
        out = self.hidden_layer_11(out)
        out = self.hidden_layer_12(out)

        out = self.output_layer(out)
        return out

    def name(self):
        return "FeedForwardRegressor12"


class VanillaUNet(NamedNeuralNetwork):
    def __init__(self, in_channels=1, out_channels=1, internal_channels=16, dropout=0,
                 skip_kernel_size=3, skip_stride=1, skip_padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 3  # 3x3 conv
        self.contract_1_1 = self.contract(in_channels, internal_channels, kernel_size, dropout=dropout)
        self.contract_1_2 = self.contract(internal_channels, internal_channels, kernel_size, dropout=dropout)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_1 = self.skip(channels=internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_2_1 = self.contract(internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.contract_2_2 = self.contract(2*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_2 = self.skip(channels=2*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_3_1 = self.contract(2*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.contract_3_2 = self.contract(4*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_3 = self.skip(channels=4*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.contract_4_1 = self.contract(4*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.contract_4_2 = self.contract(8*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_4 = self.skip(channels=8*internal_channels, skip_kernel_size=skip_kernel_size,
                                skip_stride=skip_stride, skip_padding=skip_padding, dropout=dropout)

        self.center = nn.Sequential(
            nn.Conv2d(8*internal_channels, 16*internal_channels, 3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
            self.skip(16*internal_channels, skip_kernel_size, skip_stride, skip_padding, dropout),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16*internal_channels, 8*internal_channels, kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

        self.expand_4_1 = self.expand(16*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.expand_4_2 = self.expand(8*internal_channels, 8*internal_channels, kernel_size, dropout=dropout)
        self.upscale_4 = nn.ConvTranspose2d(8*internal_channels, 4*internal_channels, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(8*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.expand_3_2 = self.expand(4*internal_channels, 4*internal_channels, kernel_size, dropout=dropout)
        self.upscale_3 = nn.ConvTranspose2d(4*internal_channels, 2*internal_channels, kernel_size=2, stride=2)

        self.expand_2_1 = self.expand(4*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.expand_2_2 = self.expand(2*internal_channels, 2*internal_channels, kernel_size, dropout=dropout)
        self.upscale_2 = nn.ConvTranspose2d(2*internal_channels, internal_channels, kernel_size=2, stride=2)

        self.expand_1_1 = self.expand(2*internal_channels, internal_channels, kernel_size, dropout=dropout)
        self.expand_1_2 = self.expand(internal_channels, internal_channels, kernel_size, dropout=dropout)
        self.final = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1),
            nn.LeakyReLU()
        )

    @staticmethod
    def contract(in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
                nn.Dropout2d(p=dropout),
                nn.LeakyReLU())

    @staticmethod
    def expand(in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2)),
            nn.Dropout2d(p=dropout),
            nn.LeakyReLU(),
        )

    @staticmethod
    def skip(channels, skip_kernel_size, skip_stride, skip_padding, dropout):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(3, skip_kernel_size),
                      stride=(1, skip_stride),
                      padding=(1, skip_padding)),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        contract_1_2 = self.contract_1_2(self.contract_1_1(x))
        pool_1 = self.pool_1(contract_1_2)
        skip_1 = self.skip_1(contract_1_2)

        contract_2_2 = self.contract_2_2(self.contract_2_1(pool_1))
        pool_2 = self.pool_2(contract_2_2)
        skip_2 = self.skip_2(contract_2_2)

        contract_3_2 = self.contract_3_2(self.contract_3_1(pool_2))
        pool_3 = self.pool_3(contract_3_2)
        skip_3 = self.skip_3(contract_3_2)

        contract_4_2 = self.contract_4_2(self.contract_4_1(pool_3))
        pool_4 = self.pool_4(contract_4_2)
        skip_4 = self.skip_4(contract_4_2)

        center = self.center(pool_4)

        concatenated_skip_4 = concatenate([center, skip_4], 1)
        expand_4_1 = self.expand_4_1(concatenated_skip_4)
        expand_4_2 = self.expand_4_2(expand_4_1)
        upscale_4 = self.upscale_4(expand_4_2)

        concatenated_skip_3 = concatenate([upscale_4, skip_3], 1)
        expand_3_1 = self.expand_3_1(concatenated_skip_3)
        expand_3_2 = self.expand_3_2(expand_3_1)
        upscale3 = self.upscale_3(expand_3_2)

        concatenated_skip_2 = concatenate([upscale3, skip_2], 1)
        expand_2_1 = self.expand_2_1(concatenated_skip_2)
        expand_2_2 = self.expand_2_2(expand_2_1)
        upscale2 = self.upscale_2(expand_2_2)

        concatenated_skip_1 = concatenate([upscale2, skip_1], 1)
        expand_1_1 = self.expand_1_1(concatenated_skip_1)
        expand_1_2 = self.expand_1_2(expand_1_1)

        output = self.final(expand_1_2)

        return output

    def name(self):
        return "UNet_in_" + str(self.in_channels) + "_out_" + str(self.out_channels)

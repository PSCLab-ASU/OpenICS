import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            #nn.BatchNorm2d(nFeat),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            #nn.BatchNorm2d(nFeat)
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        self.res = self.main(x)
        self.res += x
        self.res = self.relu(self.res)

        return self.res

class ResBlock_wBN(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(ResBlock_wBN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(nFeat),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(nFeat)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        self.res = self.main(x)
        self.res += x
        self.res = self.relu(self.res)
        return self.res

class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, leny, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.base = 64
        self.fs = 8
        self.nResBlock = 2
        self.leny = leny

        self.linear1 = nn.Linear(self.leny, self.base * self.fs ** 2) # 64x8x8
        self.bn_linear1 = nn.BatchNorm1d(self.base * self.fs ** 2)
        self.conv1 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        self.bn1 = nn.BatchNorm2d(self.base)
        #self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4

        modules = [ResBlock(self.base, 3) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)  # 3x8x8
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.relu(self.conv1(self.output))
        self.output = self.resblk1(self.output)
        self.output = self.conv2(self.output)
        self.output = self.tanh(self.output)

        return self.output

class LAPGAN_Discriminator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level1, self).__init__()

        self.base = 32
        self.channels = channels
        self.ngpu = ngpu
        self.fs = 8

        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 32x8x8
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 32x4x4
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, 2 * self.base, kernel_size=3, padding=1, stride=1, bias=False),    # 64x4x4
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            #print(input.shape)
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# fix gen1 when updating gen2
class LAPGAN_Generator_level2(nn.Module):
    def __init__(self, channels, leny, ngpu):
        super(LAPGAN_Generator_level2, self).__init__()

        self.base = 64
        self.leny = leny
        self.channels = channels
        self.ngpu = ngpu
        self.fs = 8
        self.nResBlock = 3

        self.upsamp1 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=4, padding=1, stride=2, bias=False)  # 3x16x16
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        self.bn1 = nn.BatchNorm2d(self.base)
        self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4
        self.bn2 = nn.BatchNorm2d(self.base)

        self.linear1 = nn.Linear(self.base * ((self.fs // 2) ** 2), self.leny)
        self.bn_linear1 = nn.BatchNorm1d(self.leny)
        self.linear2 = nn.Linear(self.leny * 2, self.base * (self.fs ** 2)) # _1 channel _2 base
        self.bn_linear2 = nn.BatchNorm1d(self.base * (self.fs ** 2))

        self.upsamp2 = nn.ConvTranspose2d(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False) # 64x16x16
        self.bn3 = nn.BatchNorm2d(self.base)

        modules = [ResBlock(self.base) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, input, y):
        self.output_up = self.upsamp1(input)

        self.output = self.relu(self.conv1(input))
        self.output = self.relu(self.conv2(self.output))
        self.output = self.output.view(self.output.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = torch.cat((y.view(y.size(0), -1), self.output), 1)
        self.output = self.relu(self.linear2(self.output))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.relu(self.upsamp2(self.output))
        self.output = self.resblk1(self.output)

        self.output = self.conv5(self.output)

        self.output = self.output + self.output_up
        self.output = self.tanh(self.output)

        return self.output

# The implementation of the disc2, input is 1x16x16
class LAPGAN_Discriminator_level2(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level2, self).__init__()

        self.base = 32
        self.channels = channels
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 32x16x16
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 32x8x8
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, 2 * self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 64x8x8
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 2 * self.base, kernel_size=3, padding=1, stride=2, bias=False), # 64x4x4
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 4 * self.base, kernel_size=3, padding=1, stride=1, bias=False), # 128x4x4
            #nn.BatchNorm2d(4 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.base, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class LAPGAN_Generator_level3(nn.Module):
    def __init__(self, channels, leny, ngpu):
        super(LAPGAN_Generator_level3, self).__init__()

        self.base = 64
        self.leny = leny
        self.channels = channels
        self.fs = 16
        self.nResBlock = 3

        self.upsamp1 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=4, padding=1, stride=2,
                                          bias=False)  # 3x16x16
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x16x16
        self.bn1 = nn.BatchNorm2d(self.base)
        self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x8x8
        self.bn2 = nn.BatchNorm2d(self.base)

        self.linear1 = nn.Linear(self.base * ((self.fs // 2) ** 2), self.leny)
        self.bn_linear1 = nn.BatchNorm1d(self.leny)
        self.linear2 = nn.Linear(self.leny * 2, self.channels * (self.fs ** 2))
        self.bn_linear2 = nn.BatchNorm1d(self.channels * (self.fs ** 2))

        self.upsamp2 = nn.ConvTranspose2d(self.channels, self.base, kernel_size=4, padding=1, stride=2,
                                  bias=False)  # 64x16x16
        self.bn3 = nn.BatchNorm2d(self.base)

        modules = [ResBlock(self.base) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, input, y):
        self.output_up = self.upsamp1(input)

        self.output = self.relu(self.conv1(input))
        self.output = self.relu(self.conv2(self.output))
        self.output = self.output.view(self.output.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = torch.cat((y.view(y.size(0), -1), self.output), 1)
        self.output = self.relu(self.linear2(self.output))
        self.output = self.output.view(-1, self.channels, self.fs, self.fs)
        self.output = self.relu(self.upsamp2(self.output))
        self.output = self.resblk1(self.output)

        self.output = self.conv3(self.output)

        self.output = self.output + self.output_up
        self.output = self.tanh(self.output)

        return self.output

# The implementation of the disc3, input is 3x32x32
class LAPGAN_Discriminator_level3(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level3, self).__init__()

        self.base = 32
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 32x32x32
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 32x16x16
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, 2 * self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 64x16x16
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 2 * self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 64x8x8
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 4 * self.base, kernel_size=3, padding=1, stride=1, bias=False),   # 128x8x8
            #nn.BatchNorm2d(4 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.base, 4 * self.base, kernel_size=3, padding=1, stride=2, bias=False),   # 128x4x4
            #nn.BatchNorm2d(4 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.base, 8 * self.base, kernel_size=3, padding=1, stride=1, bias=False),   # 256x4x4
            #nn.BatchNorm2d(8 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * self.base, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# orig
# The implementation of the gen4, input is 3x32x32
class LAPGAN_Generator_level4(nn.Module):
    def __init__(self, channels, leny, ngpu):
        super(LAPGAN_Generator_level4, self).__init__()

        self.base = 64
        self.leny = leny
        self.channels = channels
        self.ngpu = ngpu
        self.fs = 32
        self.nResBlock = 3

        self.upsamp1 = nn.ConvTranspose2d(self.channels, self.channels, kernel_size=4, padding=1, stride=2,
                                          bias=False)  # 3x16x16
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        self.bn1 = nn.BatchNorm2d(self.base)
        self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4
        self.bn2 = nn.BatchNorm2d(self.base)

        self.linear1 = nn.Linear(self.base * ((self.fs // 2) ** 2), self.leny)
        self.linear2 = nn.Linear(self.leny * 2, self.channels * (self.fs ** 2))

        self.upsamp2 = nn.ConvTranspose2d(self.channels, self.base, kernel_size=4, padding=1, stride=2,
                                          bias=False)  # 64x16x16
        self.bn3 = nn.BatchNorm2d(self.base)

        modules = [ResBlock(self.base) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)

        self.tanh = nn.Tanh()


    def forward(self, input, y):
        self.output_up = self.upsamp1(input)

        self.output = self.relu(self.conv1(input))
        self.output = self.relu(self.conv2(self.output))
        self.output = self.output.view(self.output.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = torch.cat((y.view(y.size(0), -1), self.output), 1)
        self.output = self.relu(self.linear2(self.output))
        self.output = self.output.view(-1, self.channels, self.fs, self.fs)
        self.output = self.relu(self.upsamp2(self.output))
        self.output = self.resblk1(self.output)

        self.output = self.conv3(self.output)

        self.output = self.output + self.output_up
        self.output = self.tanh(self.output)

        return self.output

# The implementation of the disc4, input is 1x64x64
class LAPGAN_Discriminator_level4(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level4, self).__init__()

        self.base = 32
        self.channels = channels
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 32x64x64
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 32x32x32
            #nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.base, 2 * self.base, kernel_size=3, padding=1, stride=1, bias=False),  # 64x32x32
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 2 * self.base, kernel_size=3, padding=1, stride=2, bias=False),  # 64x16x16
            #nn.BatchNorm2d(2 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.base, 4 * self.base, kernel_size=3, padding=1, stride=1, bias=False),   # 128x16x16
            #nn.BatchNorm2d(4 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.base, 4 * self.base, kernel_size=3, padding=1, stride=2, bias=False),   # 128x8x8
            #nn.BatchNorm2d(4 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.base, 8 * self.base, kernel_size=3, padding=1, stride=1, bias=False),   # 256x8x8
            #nn.BatchNorm2d(8 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * self.base, 8 * self.base, kernel_size=3, padding=1, stride=2, bias=False),   # 256x4x4
            #nn.BatchNorm2d(8 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * self.base, 16 * self.base, kernel_size=3, padding=1, stride=1, bias=False),   # 512x4x4
            #nn.BatchNorm2d(16 * self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16 * self.base, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

'''
class LAPGAN(nn.Module):
    def __init__(self, channels, ngpu, LAPGAN_Generator_level1, LAPGAN_Generator_level2, LAPGAN_Generator_level3,
                 LAPGAN_Generator_level4):
        super(LAPGAN, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.LAPGAN_Generator_level1 = LAPGAN_Generator_level1
        self.LAPGAN_Generator_level2 = LAPGAN_Generator_level2
        self.LAPGAN_Generator_level3 = LAPGAN_Generator_level3
        self.LAPGAN_Generator_level4 = LAPGAN_Generator_level4

    def forward(self, input, y2, y3, y4):
        self.output = self.LAPGAN_Generator_level1(input)
        self.output = self.LAPGAN_Generator_level2(self.output, y2)
        self.output = self.LAPGAN_Generator_level3(self.output, y3)
        self.output = self.LAPGAN_Generator_level4(self.output, y4)

        return self.output
'''


'''
# The implementation of gen1, input is 3x8x8
# first tab
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.base = 64
        self.fs = 8
        self.nResBlock = 3

        self.linear1 = nn.Linear(self.channels * self.fs ** 2, self.base * self.fs **2) # 64x8x8
        self.conv1 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        #self.bn1 = nn.BatchNorm2d(self.base)
        #self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4

        modules = [ResBlock(self.base, 3) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)  # 3x8x8
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.relu(self.conv1(self.output))
        self.output = self.resblk1(self.output)
        self.output = self.conv2(self.output)
        self.output = self.tanh(self.output)

        return self.output
'''
# second tab
'''
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.base = 64
        self.fs = 8
        self.nResBlock = 3

        self.linear1 = nn.Linear(self.channels * self.fs ** 2, self.base * self.fs **2) # 64x8x8
        self.bn1_linear = nn.BatchNorm1d(self.base * self.fs **2)
        self.conv1 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        self.bn1 = nn.BatchNorm2d(self.base)
        #self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4

        modules = [ResBlock(self.base, 3) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)  # 3x8x8
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.relu(self.bn1_linear(self.linear1(self.output)))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.relu(self.bn1(self.conv1(self.output)))
        self.output = self.resblk1(self.output)
        self.output = self.conv2(self.output)
        self.output = self.tanh(self.output)

        return self.output
'''
# tab3
'''
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.base = 64
        self.fs = 8
        self.nResBlock = 3

        self.linear1 = nn.Linear(self.channels * self.fs ** 2, self.base * self.fs **2) # 64x8x8
        self.bn1_linear = nn.BatchNorm1d(self.base * self.fs **2)
        self.conv1 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        self.bn1 = nn.BatchNorm2d(self.base)
        #self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4

        modules = [ResBlock(self.base, 3) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)  # 3x8x8
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.bn1_linear(self.relu(self.linear1(self.output)))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.bn1(self.relu(self.conv1(self.output)))
        self.output = self.resblk1(self.output)
        self.output = self.conv2(self.output)
        self.output = self.tanh(self.output)

        return self.output
'''
''' tab4
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.base = 64
        self.fs = 8
        self.nResBlock = 3

        self.linear1 = nn.Linear(self.channels * self.fs ** 2, self.base * self.fs **2) # 64x8x8
        self.conv1 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=1, bias=False)  # 64x8x8
        #self.bn1 = nn.BatchNorm2d(self.base)
        #self.conv2 = nn.Conv2d(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False)  # 64x4x4

        modules = [ResBlock(self.base, 3) for _ in range(self.nResBlock)]
        self.resblk1 = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=False)  # 3x8x8
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.relu(self.linear1(self.output))
        self.output = self.output.view(-1, self.base, self.fs, self.fs)
        self.output = self.relu(self.conv1(self.output))
        self.output = self.resblk1(self.output)
        self.output = self.conv2(self.output)
        self.output = self.tanh(self.output)

        return self.output
'''
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)     # convolutional layer with stride
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)         # convolutional layer without stride
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)                # 1*1 convolutional layer

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        down_sampled_x = self.conv3(x)
        down_sampled_x = self.bn3(down_sampled_x)
        out += down_sampled_x
        out = F.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        # input is rgb image which has 3 channels
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))     # make output size (1,1) means global average
        self.fc = nn.Linear(512, 2)                             # output_channel is equal to number of classification

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.global_avg_pool(x)             # shape should be (B,C,1,1)
        x = x.view((x.size(0), -1))             # faltten_layer: shape should be (B,C)
        x = self.fc(x)

        return x

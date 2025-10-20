import torch.nn as nn
import torch


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=1, kernel_size=3,
                     bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class Res34Net(nn.Module):

    def __init__(self, layers, block=BasicBlock, num_class=10, ram=False):
        super(Res34Net, self).__init__()
        self.ram = ram
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, block=block, num_block=layers[0])
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, block=block, num_block=layers[1], stride=2)
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, block=block, num_block=layers[2], stride=2)
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, block=block, num_block=layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

        self.cls_layer_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1)
        self.fc_relu_1 = nn.ReLU(True)
        self.cls_layer_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1)
        self.fc_relu_2 = nn.ReLU(True)
        self.cls_layer_3 = nn.Conv2d(1024, num_class, kernel_size=1, padding=0)

    def make_layer(self, block, in_channels, out_channels, num_block, stride=1, downsample=None):
        if in_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels=in_channels, out_channels=out_channels, downsample=downsample, stride=stride))
        for _ in range(num_block - 1):
            layers.append(block(in_channels=out_channels, out_channels=out_channels, downsample=None, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, out_ram=None):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # [4,512,10,10]
        out_cls = self.avg_pool(out) # [4,512,1,1]
        out_cls = out_cls.view(x.shape[0], -1)
        out_cls = self.fc(out_cls)
        if self.ram:
            out_ram = self.cls_layer_1(out)
            out_ram = self.fc_relu_1(out_ram)
            out_ram = self.cls_layer_2(out_ram)
            out_ram = self.fc_relu_2(out_ram)
            out_ram = self.cls_layer_3(out_ram)
        return out_cls, out_ram


if __name__ == "__main__":
    model = Res34Net([3, 4, 6, 3])
    print(model)
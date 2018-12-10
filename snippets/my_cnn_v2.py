import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.conv4_bn(F.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.conv6_bn(F.relu(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    model = Cnn()
    print(model)
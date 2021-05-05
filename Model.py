from torch import nn


class Model1(nn.Module):
    def __init__(self, num_classes=10):
        super(Model1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu(x)
        return out


class Model2(nn.Module):
    def __init__(self, num_classes=10):
        super(Model2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(14 * 14 * 32, num_classes)          # (((W - K + 2P)/S) + 1) : cong thuc

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

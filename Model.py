from torch import nn


class Model1(nn.Module):
    def __init__(self, num_classes=10):
        super(Model1, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.func = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.func(out)
        out = self.fc2(out)
        out = self.func(out)
        out = self.fc3(out)
        return out


class Model2(nn.Module):
    def __init__(self, num_classes=10):
        super(Model2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(14 * 14 * 32, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

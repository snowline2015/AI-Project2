from torch import nn
import torch.nn.functional as F


class Model1(nn.Module):                        # 3 Fully connected layers
    def __init__(self, num_classes=10):
        super(Model1, self).__init__()
        self.linear_leaky_relu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),            # Default MNIST size 784 = 28 * 28
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        out = self.linear_leaky_relu(x)
        return F.log_softmax(out, dim=1)


class Model2(nn.Module):                        # 2 Convolutional layers, Pooling layers, Fully connected layers
    def __init__(self, num_classes=10):
        super(Model2, self).__init__()

        self.conv = nn.Sequential(                                          # formula: (((W - K + 2P)/S) + 1)
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),           # [(28 - 5 + 2*2)/1] + 1 = 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # divide 2 (28 / 2 = 14)
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),          # [(14 - 5 + 2*2)/1] + 1 = 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                          # divide 2 (14 / 2 = 7)

        self.linear_relu = nn.Sequential(
            nn.Linear(7 * 7 * 64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),                  # Avoid over-fitting
            nn.Linear(64, num_classes))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.linear_relu(out)
        return F.log_softmax(out, dim=1)

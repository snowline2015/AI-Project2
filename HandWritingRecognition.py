import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from Model import Model1, Model2

batch_size = 32
learning_rate = 0.01
num_epochs = 20

# Data Loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ])),
    batch_size=batch_size,
    shuffle=True)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ])),
    batch_size=batch_size,
    shuffle=False)


# Visualize Data
def imshow(img, mean, std):
    img = img / std + mean  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images), 0.1307, 0.3081)
print(labels)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
mod_num = int(input("1. Fully Connected Layer\n2. Convolutional layer, Pooling layer, Fully connected layer\n"
      "Choose model to train: "))
if mod_num == 1:
    model = Model1().to(device)
else:
    model = Model2().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

num_steps = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print("Epoch {}/{} - Step: {}/{} - Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, num_steps, total_loss / (i + 1)))

    torch.save(model.state_dict(), 'model.pth')
    model.eval()

    val_losses = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for _, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_losses += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Epoch {} - Accuracy: {}% - Validation Loss : {:.4f}\n".format(
            epoch + 1,
            correct / total * 100,
            val_losses / (len(val_loader))))

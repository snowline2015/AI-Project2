import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import Function
from Model import Model1, Model2

batch_size = 32
learning_rate = 0.01
num_epochs = 20

# Data Loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)), ])),
    batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)), ])),
    batch_size=batch_size, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
mod_num = int(input("1. Fully Connected Layer\n2. Convolutional layer, Pooling layer, Fully connected layer\n"
      "Choose model to train: "))
model = Model1().to(device) if mod_num == 1 else Model2().to(device)

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

        if i % 100 == 0:
            print("Epoch {}/{} - Step: {}/{} - Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, num_steps, total_loss / (i + 1)))

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
            epoch + 1, correct / total * 100, val_losses / (len(val_loader))))

torch.save(model.state_dict(), 'test/model.pth')


model = Model1() if mod_num == 1 else Model2()
model.load_state_dict(torch.load('test/model.pth'))
model.eval()

dataiter = iter(val_loader)
images, labels = dataiter.next()

while (True):
    i = int(input("Input image index to predict (1-32, outrange to exit): "))
    if (i <= 0 or i > 32): break
    Function.imshow(images[i - 1])
    print('Label:', labels[i - 1], ', Predicted:', Function.predict_image(images[i - 1], model))


file_path = Function.filedialog.askopenfilename()

input_img = Function.prepare_image(file_path)
prediction = torch.argmax(model(input_img)).item()
print('Image predicted as ', prediction)









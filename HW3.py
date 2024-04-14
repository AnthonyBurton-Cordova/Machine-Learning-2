# THIS IS HW 3

# this is very powerful what you are about to do!!!
# Your job is to generate a homework assignment where you train a neural net with convolution to classify the first zero through four digits in Emnes.
# Then you will do transfer learning on the digits five through nine using 1/10 of the data available to you.
# Finally, you will need to generate your own data set of the letters a through E you will take pictures and import them and try to transfer to those images.


#### Question 1 - Train a CNN to classify 0-4 in MNIST

#### Import Libraries for the whole project
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader, Dataset
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


### Initial NN for training on 0-4 and freeze layers within the NN

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter dataset to include only digits 0-4
train_indices = train_dataset.targets <= 4
test_indices = test_dataset.targets <= 4

train_dataset.targets = train_dataset.targets[train_indices]
train_dataset.data = train_dataset.data[train_indices]
test_dataset.targets = test_dataset.targets[test_indices]
test_dataset.data = test_dataset.data[test_indices]

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(4*4*64, 100)
        self.fc2 = nn.Linear(100, 5)  # 5 classes for digits 0-4

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 4*4*64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Freeze layers except the final layer
for param in model.parameters():
    param.requires_grad = False

model.fc2.weight.requires_grad = True
model.fc2.bias.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Define dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training the model
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Save the trained model for transfer learning
torch.save(model.state_dict(), 'pretrained_model.pth')

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ones = x_train[y_train == 1]
twos = x_train[y_train == 2]
threes = x_train[y_train == 3]
fours = x_train[y_train == 4]
fives = x_train[y_train == 5]
sixes = x_train[y_train == 6]
sevens = x_train[y_train == 7]
eights = x_train[y_train == 8]
nines = x_train[y_train == 9]

# Calculate the average image of all the 7's
average_one = np.mean(ones, axis=0)
average_two = np.mean(twos, axis=0)
average_three = np.mean(threes, axis=0)
average_four = np.mean(fours, axis=0)
average_five = np.mean(fives, axis=0)
average_six = np.mean(sixes, axis=0)
average_seven = np.mean(sevens, axis=0)
average_eight = np.mean(eights, axis=0)
average_nine = np.mean(nines, axis=0)


plt.imshow(nines[10])

### 
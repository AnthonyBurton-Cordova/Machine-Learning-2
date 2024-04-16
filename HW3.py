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

print(f"Accuracy on test set 0-4: {(100 * correct / total):.2f}%")

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

#View MNIST numbers
plt.imshow(sevens[121])
plt.imshow(nines[10])

# Generate a weak signal
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
frequency = 5  # Frequency of the sine wave
weak_signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # Weak sine wave signal

# Add noise
noise_level = 1.3  # Adjust this to see the effect of different noise levels
noise = noise_level * np.random.randn(len(t))
noisy_signal = weak_signal + noise

# Filter the noisy signal to demonstrate stochastic resonance
# This is a very simplistic way of filtering just for demonstration purposes
filtered_signal = np.convolve(noisy_signal, np.ones((50,))/50, mode='same')

# Plotting
plt.figure(figsize=(15, 6))

plt.subplot(3, 1, 1)
plt.plot(t, weak_signal, label='Weak Signal')
plt.title('Original Weak Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.title('Weak Signal with Noise')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.title('Filtered Signal Demonstrating Stochastic Resonance')
plt.legend()

plt.tight_layout()
plt.show()



#### Question 2 - Transfer Learning on 5-9 using 1/10 of the data.

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Filter dataset to include only digits 5-9
train_indices = train_dataset.targets >= 5
test_indices = test_dataset.targets >= 5

train_dataset.targets = train_dataset.targets[train_indices] - 5
train_dataset.data = train_dataset.data[train_indices]
test_dataset.targets = test_dataset.targets[test_indices] - 5
test_dataset.data = test_dataset.data[test_indices]

# Define the number of samples to use for testing (1/10 of the available data)
num_test_samples = len(test_dataset) // 10

# Create a random subset of the test dataset
test_subset_indices = torch.randperm(len(test_dataset))[:num_test_samples]
test_subset = Subset(test_dataset, test_subset_indices)

# Ensure non-empty datasets
if len(train_dataset) == 0 or len(test_subset) == 0:
    raise ValueError("Empty dataset after filtering")

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define number of epochs
epochs = 7

# Define batch size
batch_size = 32

# Training the model
for epoch in range(epochs):
    model.train()  # Set model to training mode
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
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set (using 1/10 of the available data) for 5-9: {accuracy:.2f}%")




#### Question 3 - Generate a data set of A - E and transfer the learning to predict those images
from PIL import Image

# Load an image.
### Please dowload the images attached (A-E) to the submission and then copy and paste the local image here
A = Image.open(r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\A.jpeg')
B = Image.open(r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\B.jpeg')
C = Image.open(r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\C.jpeg')
D = Image.open(r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\D.jpeg')
E = Image.open(r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\E.jpeg')

# Resize all the images
resized_image = A.resize((500, 500))

# Display the image
resized_image.show()

# Define the new size
new_size = (500, 500)

# Paths to the images
image_paths = [
    r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\A.jpeg',
    r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\B.jpeg',
    r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\C.jpeg',
    r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\D.jpeg',
    r'C:\Users\antho\OneDrive\SMU\Semester 5 Spring 2023\DS 7335 Machine Learning II\HW 3\E.jpeg'
]
    
# Load each image, resize it, and display it
for path in image_paths:
    img = Image.open(path)
    resized_img = img.resize(new_size)
    
    # Display the image
    resized_img.show(title=os.path.basename(path))


# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Path to your folder containing images of letters A through E
root_dir = "/content/drive/My Drive/Letters"

# Create an instance of your custom dataset
custom_dataset = CustomDataset(root_dir, transform=transform)

# Define a DataLoader for your custom dataset
custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

# Load the pretrained model
pretrained_model_path = "pretrained_model.pth"
pretrained_model = CNN()
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.01)

# Define the true labels for the custom dataset
true_labels = [0, 1, 2, 3, 4]  # Corresponding to letters A through E

# Evaluate the model on your custom dataset
correct = 0
total = 0
pretrained_model.eval()
with torch.no_grad():
    for images, labels in zip(custom_loader, true_labels):
        outputs = pretrained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = (correct / total) * 100
print(f"Accuracy on custom dataset for letters A through E: {accuracy:.2f}%")



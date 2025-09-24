import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import os 
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
# Define a transform to resize images to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(), # Add this
    transforms.RandomRotation(15),     # Add this
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Add this for transfer learning
])
# Load your custom dataset
train_dir = 'data/hotdog_nothotdog/train'
test_dir = 'data/hotdog_nothotdog/test'

trainset = datasets.ImageFolder(root=train_dir, transform=transform)
testset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class MAMPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MAMPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Calculate max, min, and average pooling
        max_pool = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        min_pool = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=self.stride)
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Calculate Max - Min and Avg
        max_min_diff = max_pool - min_pool
        avg = avg_pool

        # Create a mask for the condition: Max - Min > Avg
        condition = max_min_diff > avg
        condition = condition.float()

        # Apply the MAM formula
        # Case 1: Max - Min > Avg
        output_max_min = max_pool - min_pool
        # Case 2: Max - Min <= Avg
        output_max_avg = (max_pool + avg_pool) / 2

        # Select the output based on the condition
        output = condition * output_max_min + (1 - condition) * output_max_avg

        return output


class Network(nn.Module):
    def __init__(self):

        #we are starting by 3 convolutional network, the kernel is fixed to 3 for know, it will increase if the model is not good enough.
        super(Network, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc=nn.Sequential(
            nn.Linear(50176, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,2),
        )

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf in MAMPooling input!")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with a new one for 2 classes
# The input features to the final layer can be found by inspecting the original model.
# For resnet18, it is 512.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move the modified model to the correct device
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#We define the training as a function so we can easily re-use it.
# We define the training as a function so we can easily re-use it.
def train(model, optimizer, num_epochs=10):
    def loss_fun(output, target):
        return nn.CrossEntropyLoss()(output, target)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        # Removed the outer tqdm and kept the inner one
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict

out_dict = train(model, optimizer, num_epochs=10)

output_filename = 'training_results.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(out_dict, f)

print(f"Training results saved to {output_filename}")
# ...
# ...
plt.legend(('Test error','Train eror'))
plt.xlabel('Epoch number')
plt.ylabel('Accuracy')
plt.plot(out_dict['test_acc'], label='Test error')
plt.plot(out_dict['train_acc'], label='Train eror')
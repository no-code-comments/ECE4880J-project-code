#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# MIT LICENSE
'''
Copyright (c) 2020 Reserved

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import pandas as pd
import os
from PIL import Image
from torch.autograd import Variable


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=6, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=6, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=6, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 4 * 4, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        # print(output.shape)
        output = F.relu(self.bn5(self.conv5(output)))
        # print(output.shape)
        output = output.view(-1, 24 * 4 * 4)
        output = self.fc1(output)

        return output


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class Paddy(Dataset):
    def __init__(self, files_name, transform, mode="train"):
        super(Paddy).__init__()
        self.tfm = transform
        self.mode = mode
        self.paddy_dict = {'bacterial_leaf_blight': 0, 'bacterial_leaf_streak': 1, 'bacterial_panicle_blight': 2, \
                           'blast': 3, 'brown_spot': 4, 'dead_heart': 5, 'downy_mildew': 6, 'hispa': 7, 'normal': 8,
                           'tungro': 9}
        self.files_name = files_name

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, idx):
        file_name = self.files_name[idx]
        im = self.tfm(Image.open(file_name))
        if self.mode == "train":
            label = self.paddy_dict[file_name.split("/")[-2]]
        else:
            label = -1
        return im, label


# Instantiate a neural network model
model = Network()

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

### Train_path is "paddy-disease-classification"; Test_path is "test_images"
Train_path = './'
Test_path = './test_images'

train_files_name = []
train_info = pd.read_csv(f"{Train_path}train.csv")
for index, row in train_info.iterrows():
    train_files_name.append(f"{Train_path}train_images/{row['label']}/{row['image_id']}")
random.shuffle(train_files_name)

test_files_name = []
for image in os.listdir(Test_path):
    test_files_name.append(f"{Test_path}/{image}")

# %%
N = 5
Train_set = []
Val_set = []
set_num = len(train_files_name) // N
Train_set = train_files_name[: (N - 1) * set_num]
Val_set = train_files_name[(N - 1) * set_num:]

train_data = Paddy(files_name=Train_set, transform=transform, mode="train")
val_data = Paddy(files_name=Val_set, transform=transform, mode="train")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(device):
    model.eval()
    accuracy = 0.0
    loss = 0.0
    total = 0.0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            total += images.shape[0]
            outputs = model(images)
            loss += loss_fn(outputs, labels).detach().item()
            accuracy += (outputs.argmax(dim=1).flatten() == labels.flatten()).float().sum().item()

    # compute the accuracy over all test images
    # print(loss, accuracy, total)
    loss /= total
    accuracy /= total
    return (loss, accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0
        total_num = 0

        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            num = images.shape[0]
            total_num += num

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            running_acc += (outputs.argmax(dim=1).flatten() == labels.flatten()).float().sum().item()
            #     # print every 1000 (twice per epoch)
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 50))
            #     # zero the loss
            #     running_loss = 0.0
            #     running_acc = 0.0

        print(f"Epoch {epoch + 1}: train loss {running_loss / total_num} train acc: {running_acc / total_num}")
        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        test_loss, test_accuracy = testAccuracy(device)
        print(f"Epoch {epoch + 1}: test loss {test_loss} test acc: {test_accuracy}\n")

        # we want to save the model if the accuracy is the best
        if test_accuracy > best_accuracy:
            saveModel()
            best_accuracy = test_accuracy

if __name__ == "__main__":
    train(100)

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

from matplotlib import pyplot as plt

baseline_train_loss = []
baseline_test_loss = []
baseline_train_acc = []
baseline_test_acc = []


with open("./baseline.txt", "r+") as file:
    contents = file.readlines()
    mode = "train"
    for line in contents:
        if line == "\n":
            continue

        segments = line.split(" ")
        if mode == "train":
            baseline_train_loss.append(float(segments[4]))
            baseline_train_acc.append(float(segments[7]))
            mode = "test"
        elif mode == "test":
            baseline_test_loss.append(float(segments[4]))
            baseline_test_acc.append(float(segments[7]))
            mode = "train"

# print(len(baseline_train_loss))
# print(baseline_train_loss)

baseline_train_loss = [t / 651 * 10407 for t in baseline_train_loss]
baseline_test_loss = [t / 651 * 10407 for t in baseline_test_loss]

resnet_train_loss = []
resnet_test_loss = []
resnet_train_acc = []
resnet_test_acc = []

with open("record.txt", "r+") as file:
    contents = file.readlines()
    for line in contents[1:]:
        if line == "\n":
            continue

        segments = line.split(" ")
        if line.startswith("Processing"):
            continue
        elif line.startswith("Train loss"):
            resnet_train_loss.append(float(segments[2]))
        elif line.startswith("Train acc"):
            resnet_train_acc.append(float(segments[2]))
        elif line.startswith("Val loss"):
            resnet_test_loss.append(float(segments[2]))
        elif line.startswith("Val acc"):
            resnet_test_acc.append(float(segments[2]))

print(len(resnet_test_loss))
print(resnet_test_loss)

plt.style.use("seaborn")
plt.figure(figsize=(12, 8), dpi=400)
plt.rcParams["font.family"] = "Century Schoolbook"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.unicode_minus'] = False


x1 = list(range(1, 51))
x2 = list(range(1, 101))
y1 = baseline_train_loss
y2 = baseline_test_loss
z1 = resnet_train_loss[:100]
z2 = resnet_test_loss[:100]

plt.plot(x1, y1, r"r-", label="baseline train loss")
plt.plot(x1, y2, r"r--", label="baseline validation loss")
plt.plot(x2, z1, r"b-", label="our ResNet train loss")
plt.plot(x2, z2, r"b--", label="our ResNet validation loss")
plt.legend(fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"Epochs", fontsize=20)
plt.ylabel(r"Cross Entropy Loss", fontsize=20)

plt.savefig("./loss_cmp_f.png")

plt.cla()

x1 = list(range(1, 51))
x2 = list(range(1, 101))
y1 = [t * 100 for t in baseline_train_acc]
y2 = [t * 100 for t in baseline_test_acc]
z1 = [t * 100 for t in resnet_train_acc[:100]]
z2 = [t * 100 for t in resnet_test_acc[:100]]

plt.plot(x1, y1, r"r-", label="baseline train accuracy")
plt.plot(x1, y2, r"r--", label="baseline validation accuracy")
plt.plot(x2, z1, r"b-", label="our ResNet train accuracy")
plt.plot(x2, z2, r"b--", label="our ResNet validation accuracy")
plt.legend(fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"Epochs", fontsize=20)
plt.ylabel(r"Accuracy (%)", fontsize=20)

plt.savefig("./acc_cmp_f.png")

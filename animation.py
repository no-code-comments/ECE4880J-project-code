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
from matplotlib import animation

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

plt.figure(figsize=(20, 8), dpi=400)
fig, a = plt.subplots()



x1 = list(range(1, 51))
x2 = list(range(1, 101))
y1 = baseline_train_loss
y2 = baseline_test_loss
z1 = resnet_train_loss[:100]
z2 = resnet_test_loss[:100]

# x1 = list(range(1, 51))
# x2 = list(range(1, 101))
# y1 = [t * 100 for t in baseline_train_acc]
# y2 = [t * 100 for t in baseline_test_acc]
# z1 = [t * 100 for t in resnet_train_acc[:100]]
# z2 = [t * 100 for t in resnet_test_acc[:100]]

def animated(i):
    a.cla()
    plt.xlim([0, 100])
    plt.ylim([0, 2.4])

    a.plot(x1[:i], y1[:i], r"r-", label="baseline train loss")
    a.plot(x1[:i], y2[:i], r"r--", label="baseline validation loss")
    a.plot(x2[:i], z1[:i], r"b-", label="our ResNet train loss")
    a.plot(x2[:i], z2[:i], r"b--", label="our ResNet validation loss")
    plt.legend(fontsize=15, loc=1)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"Epochs", fontsize=20)
    plt.ylabel(r"Cross Entropy Loss", fontsize=20)


ani = animation.FuncAnimation(fig, animated, frames=range(100), repeat=False, interval=100)
ani.save("test2.gif", fps=30)
plt.show()

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

from torchvision import transforms
from PIL import Image

# used for report
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

t1 = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
t2 = transforms.Compose([transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()])
t3 = transforms.RandomRotation(180)
t4 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
to_image = transforms.ToPILImage()

from matplotlib import pyplot as plt
file_name = f"train_images/hispa/100003.jpg"
im = Image.open(file_name)
plt.imshow(im)
plt.savefig("DA0.png")

out = t1(im)
plt.cla()
plt.imshow(out)
plt.savefig("DA1.png")

out = t2(out)
plt.cla()
plt.imshow(out)
plt.savefig("DA2.png")

out = t3(out)
plt.cla()
plt.imshow(out)
plt.savefig("DA3.png")

out = t4(out)
plt.cla()
plt.imshow(to_image(out))
plt.savefig("DA4.png")

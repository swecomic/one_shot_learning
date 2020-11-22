'''
    This class works under the directory structure of the Omniglot Dataset
    It creates the pairs of images for inputs, same character label = 1, vice versa
'''
from os import walk
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import random



# setting the root directories and categories of the images
# root_dir = './images_background/'
root_dir = './images_evaluation/'
categories = [[folder, os.listdir(root_dir + folder)] for folder in os.listdir(root_dir)  if not folder.startswith('.') ]


# creating the pairs of images for inputs, same character label = 1, vice versa
class OmniglotDataset(Dataset):
    def __init__(self, categories, root_dir, setSize, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize

    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        img1 = None
        img2 = None
        label = None

        if idx % 2 == 0: # select the same character for both images
            category = random.choice(categories)
            character = random.choice(category[1])
            imgDir = root_dir + category[0] + '/' + character
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = Image.open(imgDir + '/' + img1Name)
            img2 = Image.open(imgDir + '/' + img2Name)
            # print(imgDir+'/'+img1Name)
            # print(imgDir+'/'+img2Name)
            label = 1.0

        else: # select a different character for both images
            category1, category2 = random.choice(categories), random.choice(categories)
            character1, character2 = random.choice(category1[1]), random.choice(category2[1])
            imgDir1, imgDir2 = root_dir + category1[0] + '/' + character1, root_dir + category2[0] + '/' + character2
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            while img1Name == img2Name:
                img2Name = random.choice(os.listdir(imgDir2))
            label = 0.0
            img1 = Image.open(imgDir1 + '/' + img1Name)
            img2 = Image.open(imgDir2 + '/' + img2Name)
#         plt.imshow(img1)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))


    '''
        This class works under the directory structure of the Omniglot Dataset
        It creates the pairs of images for inputs, same character label = 1, vice versa
    '''


# creates n-way one shot learning evaluation
class NWayOneShotEvalSet(Dataset):
    def __init__(self, categories, root_dir, setSize, numWay, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.setSize = setSize
        self.numWay = numWay
        self.transform = transform

    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        # find one main image
        category = random.choice(categories)
        character = random.choice(category[1])
        imgDir = root_dir + category[0] + '/' + character
        imgName = random.choice(os.listdir(imgDir))
        mainImg = Image.open(imgDir + '/' + imgName)
        # print(imgDir + '/' + imgName)
        if self.transform:
            mainImg = self.transform(mainImg)

        # find n numbers of distinct images, 1 in the same set as the main
        testSet = []
        label = np.random.randint(self.numWay)
        for i in range(self.numWay):
            testImgDir = imgDir
            testImgName = ''
            if i == label:
                testImgName = random.choice(os.listdir(imgDir))
            else:
                testCategory = random.choice(categories)
                testCharacter = random.choice(testCategory[1])
                testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                while testImgDir == imgDir:
                    testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                testImgName = random.choice(os.listdir(testImgDir))
            testImg = Image.open(testImgDir + '/' + testImgName)

            if self.transform:
                testImg = self.transform(testImg)
            testSet.append(testImg)

        # plt.imshow()
        return mainImg, testSet, torch.from_numpy(np.array([label], dtype=int))



# choose a training dataset size and further divide it into train and validation set 80:20
dataSize = 10000 # self-defined dataset size
TRAIN_PCT = 0.8 # percentage of entire dataset for training
train_size = int(dataSize * TRAIN_PCT)
val_size = dataSize - train_size

transformations = transforms.Compose(
    [transforms.ToTensor()])

omniglotDataset = OmniglotDataset(categories, root_dir, dataSize, transformations)
train_set, val_set = random_split(omniglotDataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)


# create the test set for final testing
testSize = 5000
numWay = 20
test_set = NWayOneShotEvalSet(categories, root_dir, testSize, numWay, transformations)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, num_workers = 0, shuffle=True)

# showing a sample input of a training set
count0 = 0
count1 = 0
for img1, img2, label in train_loader:
    print()
    if label[0] == 1.0:
        print(img1[0])
        plt.subplot(1,2,1)
        plt.imshow(img1[0][0])
        plt.subplot(1,2,2)
        plt.imshow(img2[0][0])
        # print(label)
        break
    # break

# showing a sample input of the testing set
count = 0
for mainImg, imgset, label in test_loader:
    # print(len(imgset))
    # print(label)
    # print(imgset.shape)
    if label != 1:
        for count, img in enumerate(imgset):
          plt.subplot(1, len(imgset)+1, count+1)
          plt.imshow(img[0][0])
          # print(img.shape)
        print(mainImg.shape)
        plt.subplot(1, len(imgset)+1, len(imgset)+1)
        plt.imshow(mainImg[0][0])
        count += 1
        break
    # break
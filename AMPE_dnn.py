import torch.nn as nn
import torch.nn.functional as F
import argparse


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


PARENT_PATH = './weights2/'


class My_DNN(nn.Module):
    def __init__(self, epochs=15, file='my_weights.pt'):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

        self.epochs = epochs
        self.file = file
        # self.path_train = path_train
        # self.path_test = path_test

        self.file = PARENT_PATH + self.file
        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


    def train(self):
        trainloader, valloader = before()
        train(model=self, trainloader=trainloader, valloader=valloader)

        
        

def before():

    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True,transform=transform)

    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    return (trainloader, valloader)


def train(model : My_DNN, trainloader, valloader):

    valloader = before()
    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])


    # PASO 3: DEFINICION DE LA FUNCION DE PERDIDA
    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL loss

    print('Before backward pass: \n', model.fc1.weight.grad)
    loss.backward()
    print('After backward pass: \n', model.fc1.weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()

    for e in range(model.epochs):
        running_loss = 0

        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    print("\nTraining Time (in minutes) =",(time()-time0)/60)



    # COMPROBACION Y EVALUACION DE LA PRECISION DE LA RED 
    images, labels = next(iter(valloader))
    img = images[0].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])

    print("Predicted Digit =", probab.index(max(probab)))
    print(labels[0].item())




    # 5.5

    correct_count, all_count = 0, 0
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            if (true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


    torch.save(model.state_dict(), model.file)


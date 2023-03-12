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


PARENT_PATH = './weights/'


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
        trainloader, valloader = self.__before()
        self.__train(trainloader=trainloader, valloader=valloader)

    def validate(self):
        _, valloader = self.__before()
        self.__validate(valloader=valloader)

    def inferencia(self, image):
        # COMPROBACION Y EVALUACION DE LA PRECISION DE LA RED 
        st = time()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))])
        imagen_preprocesada = transform(image)

        img = imagen_preprocesada.view(1, 784)

        with torch.no_grad():
            logps = self(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        
        et = time()
        elapsed_time = et - st

        print("Predicted Digit =", probab.index(max(probab)))
        print("Elapsed time = ", elapsed_time)




        
        

    def __before(self):

        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),])

        trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True,transform=transform)

        valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()

        return (trainloader, valloader)


    def __train(self, trainloader, valloader):

        model = self

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

        loss.backward()

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

        print("time: ",(time()-time0)/60)

        torch.save(model.state_dict(), model.file)



    def __validate(self, valloader):

        model = self

        # COMPROBACION Y EVALUACION DE LA PRECISION DE LA RED 
        images, labels = next(iter(valloader))
        img = images[0].view(1, 784)

        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])




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

        # print("Number Of Images Tested =", all_count)
        print(str(correct_count/all_count), end="")



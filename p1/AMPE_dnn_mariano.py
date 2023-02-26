import torch.nn as nn
import torch.nn.functional as F
import argparse


import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim



EPOCHS = 15
WEIGHTS_PATH='my_weights.pt'


class My_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


    def train(self):
        pass

    def inferencia(self, image):
        _,valloader = before()
        inferencia(model=self,valloader=valloader,image=image)

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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, required=False)
    parser.add_argument("-f", "--file", help="weights file", required=False)
    args = parser.parse_args()

    if args.epochs:
        EPOCHS = args.epochs

    if args.file:
        WEIGHTS_PATH = args.file

    WEIGHTS_PATH = "./weights/" + WEIGHTS_PATH


    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True,transform=transform)

    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)

    plt.imshow(images[0].numpy().squeeze(), cmap='plasma')
    plt.show()


    model = My_DNN()
    print(model)

    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))

    # Ver imagen
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    plt.tight_layout()

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='plasma')
    ax1.axis('off')

    ax2.barh(np.arange(len(ps)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(ps)))
    ax2.set_yticklabels(np.arange(len(ps)))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.show()


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

    for e in range(EPOCHS):
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


def inferencia(model : My_DNN,image):
    # COMPROBACION Y EVALUACION DE LA PRECISION DE LA RED 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])
    imagen_preprocesada = transform(image)

    img = imagen_preprocesada.view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    
    print("Predicted Digit =", probab.index(max(probab)))







if __name__ == "__main__":
    main()

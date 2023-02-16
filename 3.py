import os
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse


import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

from AMPE_dnn_mariano import My_DNN


def inferenciaMultiple(pesos_path):
   
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    # Cargar el subconjunto de validación del dataset MNIST
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    model = My_DNN()
    
    model = torch.load(pesos_path)
    # Seleccionar aleatoriamente 10 imágenes del subconjunto de validación
    imagenesTest = random.sample(range(len(valset)), 10)

    # Para cada imagen seleccionada, llamar a la función 'inferencia' y guardar el resultado
    nombre_archivo = f"resultado_ej3.txt"
    with open(nombre_archivo, 'w') as f:
        pass
    for i, indice in enumerate(imagenesTest):
        images, labels = next(iter(valloader))
        img = images[0].view(1, 784)

        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])

        print("Predicted Digit =", probab.index(max(probab)))
        print(labels[0].item())

def main():          
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="weights file", required=True)
    args = parser.parse_args()

    if args.file:
        WEIGHTS_PATH = args.file

    WEIGHTS_PATH = "./weights/" + WEIGHTS_PATH

    inferenciaMultiple(WEIGHTS_PATH)


    
if __name__ == "__main__":
    main()
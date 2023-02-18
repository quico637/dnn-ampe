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
from torchvision.transforms.functional import to_pil_image
from torch import nn, optim

from AMPE_dnn import My_DNN

PARENT_PATH = './outputs/'


def borrar_archivos_carpeta(folder_path):
    print("limpiando la carpeta de fotos")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                #print(f"Se eliminó el archivo: {file_path}")
        except Exception as e:
            print(f"Error al eliminar {file_path}: {e}")

def inferenciaMultiple(pesos_path):
    IMAGES_PATH = "./images/"
    borrar_archivos_carpeta(IMAGES_PATH)
    nombre_archivo = f"{PARENT_PATH}resultado_ej3.txt"
    with open(nombre_archivo, 'w') as f:
        pass
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

    # Cargar el subconjunto de validación del dataset MNIST
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    imagenesTest = random.sample(range(len(valset)), 10)

    for i in imagenesTest:
        image, label = valset[i]
        # Convertir el tensor a una imagen de PIL
        image = to_pil_image(image)
        # Guardar la imagen
        image.save(f"{IMAGES_PATH}/imagen_{i}.jpg")
        print(f"digitio a predicir: {label}")
        os.system(f"python3 inferencia.py -i {IMAGES_PATH}/imagen_{i}.jpg -f {pesos_path}")


def main():          
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="directory", required=True)
    args = parser.parse_args()

    if args.weights:
        WEIGHTS_PATH = args.weights

    inferenciaMultiple(WEIGHTS_PATH)


    
if __name__ == "__main__":
    main()
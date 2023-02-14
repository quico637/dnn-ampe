import argparse
import random
import sys
import torch
from PIL import Image

def inferencia(imagen_path, pesos_path):
    # Cargar la imagen
    with Image.open(imagen_path) as imagenDigito:
        imagen = imagenDigito.convert('L').resize((28,28))
    
    # Cargar los pesos
    model = torch.load(pesos_path)

    # Predecir el dígito
    with torch.no_grad():
        predictions = model(imagen.view(1,1,28,28))
        digit = torch.argmax(predictions)

    # Imprimir resultado
    print("La imagen representa el dígito:", digit.item())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="imagen para inferir", required=False)
    parser.add_argument("-f", "--file", help="weights file", required=False)
    args = parser.parse_args()

    if args.image:
        image = args.image

    if args.file:
        WEIGHTS_PATH = args.file

    WEIGHTS_PATH = "./weights/" + WEIGHTS_PATH

    inferencia(image, WEIGHTS_PATH)

if __name__ == "__main__":
    main()

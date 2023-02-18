import argparse
from AMPE_dnn import My_DNN
import torch
from PIL import Image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="imagen para inferir", required=True)
    parser.add_argument("-f", "--file", help="weights file", required=True)
    args = parser.parse_args()

    WEIGHTS_PATH = args.file

    with Image.open(args.image) as imagenDigito:
        imagen = imagenDigito.convert('L').resize((28,28))

        # Grayscale
        imagen = imagen.convert('L')
        # Threshold
        imagen = imagen.point( lambda p: 255 if p > 128 else 0 )
        # To mono
        imagen = imagen.convert('1')

        imagen.save("geeks.jpg")

        model2 = My_DNN()
        model2.load_state_dict(torch.load(WEIGHTS_PATH))
        model2.inferencia(imagen)
    

if __name__ == "__main__":
    main()

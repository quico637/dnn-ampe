import argparse
import random
import sys
from AMPE_dnn_mariano import My_DNN
import torch
from PIL import Image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="imagen para inferir", required=True)
    parser.add_argument("-f", "--file", help="weights file", required=True)
    args = parser.parse_args()

    WEIGHTS_PATH = "./weights/" + args.file

    with Image.open(args.image) as imagenDigito:
        imagen = imagenDigito.convert('L').resize((28,28))
        px = list(imagen.getdata())
        #print(px[0])

        model2 = My_DNN()
        model2.load_state_dict(torch.load(WEIGHTS_PATH))
        model2.inferencia(imagen)
    

if __name__ == "__main__":
    main()

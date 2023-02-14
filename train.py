import argparse
import os

from AMPE_dnn import My_DNN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)
    args = parser.parse_args()

    model = My_DNN(epochs=args.epochs, file=f"pesos-{args.epochs}.pt")
    model.train()

if __name__ == '__main__':
    main()

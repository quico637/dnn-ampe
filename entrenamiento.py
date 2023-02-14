
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)
args = parser.parse_args()


os.system(f"python3 ./AMPE_dnn.py -e {args.epochs} -f pesos-{args.epochs}.pt")

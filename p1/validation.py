import argparse
import torch

from AMPE_dnn import My_DNN


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="weights path", type=str, required=True)
    args = parser.parse_args()

    model2 = My_DNN()
    model2.load_state_dict(torch.load(args.file))
    model2.validate()

if __name__ == '__main__':
    main()
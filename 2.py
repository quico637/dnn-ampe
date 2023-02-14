import os
import argparse


FILE = 'vals.txt'

os.system(f'echo -n "" > {FILE}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="directory of weights", type=str, required=True)
    args = parser.parse_args()

    for file in os.listdir(args.dir):
        os.system(f"python3 validation.py -f {file} >> {FILE}")


    
if __name__ == "__main__":
    main()

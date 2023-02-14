import os
import argparse


FILE = 'accuracy.csv'

os.system(f'echo -n "" > {FILE}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="directory of weights", type=str, required=True)
    args = parser.parse_args()


    dir = args.dir

    if not args.dir[-1] == '/':
        dir += '/'

    for i in [1, 10, 20, 80]:
        n = n - 1
        os.system(f"echo -n \"{i} epochs\" >> {FILE}")
        if n > 0:
            os.system(f"echo -n \";\" >> {FILE}")

    os.system(f"echo  \"\" >> {FILE}")


    n = len(os.listdir(args.dir))
    for file in os.listdir(args.dir):
        n = n - 1
        os.system(f"python3 validation.py -f {dir}{file} >> {FILE}")

        if n > 0:
            os.system(f"echo -n \";\" >> {FILE}")




    
if __name__ == "__main__":
    main()

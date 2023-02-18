
import os

PARENT_PATH = './output/'
FILE = PARENT_PATH + 'times.csv'

def main():
    os.system(f'echo -n "" > {FILE}')


    numbers = [1, 10, 20, 80]
    n = len(numbers)
    for i in numbers:
        n = n - 1
        os.system(f"echo -n \"{i} epochs\" >> {FILE}")
        if n > 0:
            os.system(f"echo -n \";\" >> {FILE}")

    os.system(f"echo  \"\" >> {FILE}")

    n = len(numbers)
    for i in numbers:
        n = n - 1
        os.system(f"python3 train.py -e {i} >> {FILE}")

        if n > 0:
            os.system(f"echo -n \";\" >> {FILE}")

if __name__ == "__main__":
    main()
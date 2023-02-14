
import os

FILE = 'times.txt'

os.system(f'echo -n "" > {FILE}')

for i in [1, 10, 20, 80]:
    os.system(f"python3 train.py -e {i} >> {FILE}")

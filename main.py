# python imports
import numpy as np
# project imports
from network.network import NeuralNetwork



# my personal configs
MAIN_DIR = 'examples'
IN_FILES = 'in.txt'
OT_FILES = 'out.txt'
# dataset (and, or, xor)
DATA_SET = 'xor'


if __name__ == "__main__":
    # dataset
    x = []
    y = []

    # loading dataset X
    path = './' + MAIN_DIR + '/' + DATA_SET + '/'
    with open(path + IN_FILES) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            if line == "" or line == "\n":
                continue
            parts = line.split(' ')
            temp = []
            temp.append(int(parts[0]))
            temp.append(int(parts[1]))
            x.append(temp)
    # loading dataset labels
    with open(path + OT_FILES) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            if line == "" or line == "\n":
                continue
            y.append(int(line))
    
    print(x, y)

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
    X = []
    labels = []

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
            X.append(temp)
    # loading dataset labels
    with open(path + OT_FILES) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            if line == "" or line == "\n":
                continue
            labels.append(int(line))
    
    X, labels = np.asarray(X), np.asarray(labels)
    print(len(X), len(labels))

    # test and train data
    x_train, y_train = X[: int(len(X) / 2)], labels[: int(len(labels) / 2)]
    x_test, y_test = X[int(len(X) / 2) :], labels[int(len(labels) / 2) :]

    # define our 2-2-1 neural network and train it
    nn = NeuralNetwork([2, 2, 1], alpha=0.5)
    nn.fit(x_train, y_train)

    # now that our network is trained, loop over the XOR data points
    for (x, target) in zip(x_test, y_test):
        # make a prediction on the data point and display the result
        # to our console
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print(f'[INFO] data={x}, ground-truth={target}, pred={pred}, step={step}')

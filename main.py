# python imports
import numpy as np
# project imports
from network.network import NeuralNetwork



if __name__ == "__main__":
    # input size of our network
    ar = [6, 14, 2]

    # creating a network
    nn = NeuralNetwork(size=ar)

import numpy as np



"""
NeuralNetwork class
    creates a new NN with a given size.
"""
class NeuralNetwork:
    """
    constructor
        @argument input_size is the size of our neural network
    """
    def __init__(self, size: list):
        # alocating the network size
        self.size = size
        # creating our network architecture
        self.parameters = self.__build_network()
    
    """
    build network
        @returns parameters of our network architecture
    """
    def __build_network(self) -> dict:
        parameters = {}

        for le in range(1, len(self.size)):  # number of layers in the network
            parameters['W' + str(le)] = np.random.normal(
                size=(self.size[le], self.size[le - 1])
            )
            parameters['b' + str(le)] = np.zeros((self.size[le], 1))

        return parameters
    
    def activation_function(self, x):
        return 1.0 / (1 + np.exp(-x))

    def feed_forward(self):
        pass 

    def back_propagetion(self):
        pass 
    
    def error(self):
        pass

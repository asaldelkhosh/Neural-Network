import numpy as np



"""
NeuralNetwork class
    creates a new NN with a given size.
"""
class NeuralNetwork:
    """
    constructor
        @argument input_size is the size of our neural network
        @argument activation_type is the activation function type of our preseptrons
    """
    def __init__(self, size: list, activation_type='sigmoid'):
        # alocating the network size
        self.size = size
        # setting the activation type
        self.activation_type = activation_type
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
    
    """
    activation functions
        @argument x as input
        @returns output based on the activation type
            types: unit_step, sigmoid, relu, default
    """
    def __activation_function(self, x):
        if self.activation_type == 'unit_step':
            return np.heaviside(x, 1)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif self.activation_type == 'relu':
            return x * (x > 0)
        else:
            return x
    
    """
    linear_activation_forward:
        using the activation function to perform a forwarding in
        feedforward steps.
        @argument a_prev as previous answers
        @argument w as the weights
        @argument b as the baios
    """
    def __linear_activation_forward(self, a_prev, w, b):
        return self.activation((w @ a_prev) + b)

    def feed_forward(self):
        pass 

    def back_propagetion(self):
        pass 
    
    def error(self):
        pass

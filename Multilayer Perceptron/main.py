import numpy as np
import math


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]

        self.weights = []

        # generating random weights

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])  # (number of rows, number of cols)
            self.weights.append(w)

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        activations = inputs  # For the first layer

        for w in self.weights:
            h = np.dot(activations, w)  # Matrix multiplication
            activations = self.sigmoid(h)

        return activations


if __name__=='__main__':

    mlp= MLP()
    inputs= np.random.rand(mlp.num_inputs) # A vector
    outputs= mlp.forward_propagate(inputs)

    print("The inputs are", inputs)
    print("The outputs are", outputs)

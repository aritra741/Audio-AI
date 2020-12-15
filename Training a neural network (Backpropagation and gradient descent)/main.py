import numpy as np
from random import random


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]

        weights = []

        # generating random weights

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])  # (number of rows, number of cols)
            weights.append(w)

        self.weights = weights;

        activations = []

        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        derivatives = []

        for i in range(len(layers) - 1):
            a = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(a)

        self.derivatives = derivatives

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        activations = inputs  # For the first layer
        self.activations[0] = activations  # For the first layer

        for i, w in enumerate(self.weights):
            # print("activ. and w. ", activations.shape, w.shape)
            h = np.dot(activations, w)  # Matrix multiplication
            activations = self.sigmoid(h)
            self.activations[i + 1] = activations

        return activations

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def back_propagate(self, error):
        # dE/dW_i= ((y-a[i+1])*s'(h[i+1]))a_i
        # error= y-a[i+1]
        # s'(h[i+1])= s(h[i+1])( 1-s(h[i+1]) )
        # s(h[i+1])= a_i

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self.sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            # print("derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate=1):
        for i in range(len(self.weights)):
            # weights = self.weights[i]
            derivatives = self.derivatives[i]
            self.weights[i] += derivatives * learning_rate

    def mse(self, target, output):  # Mean Squared Error
        return np.average((target - output) ** 2)

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_err = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forward_propagate(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)

                sum_err += self.mse(target, output)

            print("Error: {} at epoch {}".format(sum_err / len(inputs), i))


if __name__ == '__main__':
    mlp = MLP(num_inputs=2, num_hidden=[5], num_outputs=1)

    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    mlp.train(items, targets, 100, 1);

    inputs = np.array([0.5, 0.4])
    target = np.array([0.9])

    output = mlp.forward_propagate(inputs=inputs)

    print("We predict that the sum of 0.5 and 0.4 is {}".format(output))

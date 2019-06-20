from Matrix import Matrix
import math
import random

class NeuralNetwork:

    # A Constructor initializing input, hidden and corresponding weight matrices
    def __init__(self, inputs, hidden, outputs):
        self.input_nodes = inputs
        self.hidden_nodes = hidden
        self.output_nodes = outputs

        self.weight_hi = Matrix(hidden, inputs)
        self.weight_oh = Matrix(outputs, hidden)
        self.weight_hi.randomize()
        self.weight_oh.randomize()
        self.guess = Matrix(outputs, 1)

        self.bias_ih = Matrix(hidden, 1)
        self.bias_oh = Matrix(outputs, 1)
        self.bias_ih.randomize()
        self.bias_oh.randomize()

        self.learning_rate = 0.05


    # Activation Function - Sigmoid
    @staticmethod
    def sigmoid(x):
        fire = 1 /float(1.0 + math.exp(-x))
        return fire

    @staticmethod
    def dsigmoid(y):
        return y * (1 - y)

    # Feeds the input throughout the network
    def feedforward(self, inputs):
        print("\nInputs : " + str(inputs))

        input = Matrix.fromArray(inputs)

        # Obtain the Output from the first layer
        # This will be the input to the hidden layer
        hidden_op = self.weight_hi.multiply(input)
        # hidden_op.display()

        # Add the bias
        hidden_op.add(self.bias_ih)

        # Pass through the sigmoid activation function
        hidden_op = Matrix.map(hidden_op, self.sigmoid)

        # Calculate final output
        final_op = self.weight_oh.multiply(hidden_op)

        # Add bia to the output
        final_op.add(self.bias_oh)

        # Pass through sigmoid activation function
        final_op = Matrix.map(final_op, self.sigmoid)

        print("Output : ")
        final_op.display()

    def train(self, inputs, outputs):
        input = Matrix.fromArray(inputs)

        # Obtain the Output from the first layer
        # This will be the input to the hidden layer
        hidden_ip = self.weight_hi.multiply(input)
        # hidden_op.display()

        # Add the bias
        hidden_ip.add(self.bias_ih)

        # Pass through the sigmoid activation function
        hidden_op = Matrix.map(hidden_ip, self.sigmoid)

        # Calculate final output
        final_op = self.weight_oh.multiply(hidden_op)

        # Add bia to the output
        final_op.add(self.bias_oh)

        # Pass through sigmoid activation function
        final_op = Matrix.map(final_op, self.sigmoid)

        output = Matrix.fromArray(outputs)

        # Error of the last layer
        error = Matrix.subtract_stat(output, final_op)

        # Derivative of sigmoid , applied to all elements of the matrix
        gradient = Matrix.map(final_op, self.dsigmoid)
        gradient = gradient.element_wise_multiply(error)
        gradient.scalar(self.learning_rate)

        self.bias_oh.add(gradient)

        hidden_op_t = hidden_op.transpose()
        weight_deltas = gradient.multiply(hidden_op_t)

        self.weight_oh.add(weight_deltas)

        who_t = self.weight_oh.transpose()
        hidden_error = who_t.multiply(error)

        hidden_gradient = Matrix.map(hidden_op, self.dsigmoid)
        hidden_gradient = hidden_gradient.element_wise_multiply(hidden_error)
        hidden_gradient.scalar(self.learning_rate)

        self.bias_ih.add(hidden_gradient)

        input_t = input.transpose()
        weight_deltas_ih = hidden_gradient.multiply(input_t)

        self.weight_hi.add(weight_deltas_ih)
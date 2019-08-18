from Matrix.Matrix import Matrix
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

        self.learning_rate = 0.1


    # Activation Function - Sigmoid
    @staticmethod
    def sigmoid(x):
        fire = 1 /float(1.0 + math.exp(-x))
        return fire

    @staticmethod
    def dsigmoid(y):
        der = y * (1 - y)
        return der

    # Feeds the input throughout the network
    def feedforward(self, inputs):

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

        op = Matrix.toArray(final_op)
        return op

    def train(self, inputs, outputs, lr):
        self.learning_rate = lr
        input = Matrix.fromArray(inputs)

        # Obtain the Output from the first layer
        # This will be the input to the hidden layer
        hidden_ip = self.weight_hi.multiply(input)

        # Add bias to it
        hidden_ip.add(self.bias_ih)

        # Sigmoid activation function
        hidden_op = Matrix.map(hidden_ip, self.sigmoid)

        # Obtain Final Output of NN
        final_op = self.weight_oh.multiply(hidden_op)

        # Add Bias into it
        final_op.add(self.bias_oh)

        # Sigmoid Activation function
        final_op = Matrix.map(final_op, self.sigmoid)

        # fetch Output
        output = Matrix.fromArray(outputs)
        output = output.transpose()

        # Calculate error
        error = Matrix.subtract_stat(output, final_op)

        # Calculate Gradient
        gradient = Matrix.map(final_op, self.dsigmoid)
        gradient = gradient.element_wise_multiply(error)
        gradient.scalar(self.learning_rate)
        self.bias_oh.add(gradient)

        hidden_op_t = hidden_op.transpose()

        # Calculate Delta Weights
        weight_deltas = gradient.multiply(hidden_op_t)

        # Update Current Weights
        self.weight_oh.add(weight_deltas)
        who_t = self.weight_oh.transpose()

        error = error.transpose()
        hidden_error = who_t.multiply(error)

        # Calculate Gradient between Hidden Layer and Input Layer Weights
        hidden_gradient = Matrix.map(hidden_op, self.dsigmoid)
        hidden_gradient = hidden_gradient.element_wise_multiply(hidden_error)
        hidden_gradient.scalar(self.learning_rate)

        self.bias_ih.add(hidden_gradient)

        input_t = input.transpose()

        # Calculate delta weights
        weight_deltas_ih = hidden_gradient.multiply(input_t)

        # Update Current weights
        self.weight_hi.add(weight_deltas_ih)

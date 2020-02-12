from Matrix.Matrix import Matrix
import math
import numpy as np

class NeuralNetwork:

    def __init__(self, inputs, hidden, outputs):

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
        self.sum = 0
        self.max = 0


    def softmax(self, x):
        exp = np.exp(x)
        ans = exp/self.sum
        return ans


    def dsoftmax(self, x):
        return x * (1 - x)


    @staticmethod
    def ReLU(x):
        if x > 0:
            return x
        else:
            return 0.1 * x


    @staticmethod
    def dReLU(y):
        if y > 0:
            return 1
        else:
            return 0.1



    @staticmethod
    def sigmoid(x):
        fire = 1 /float(1.0 + np.exp(-x))
        return fire


    @staticmethod
    def dsigmoid(y):
        der = y * (1 - y)
        return der



    def feedforward(self, inputs):
        self.sum = 0
        self.max = 0
        input = Matrix.fromArray(inputs)


        hidden_op = self.weight_hi.multiply(input)
        hidden_op.add(self.bias_ih)
        hidden_op = Matrix.map(hidden_op, self.sigmoid)

        final_op = self.weight_oh.multiply(hidden_op)
        final_op.add(self.bias_oh)
        final_op = Matrix.map(final_op, self.scale) 

        for i in range(final_op.rows):
            for j in range(final_op.columns):
                self.sum += np.exp(final_op.matrix[i][j])

        final_op = Matrix.map(final_op, self.softmax)
        op = Matrix.toArray(final_op)
        return op


    def scale(self, x):
        return x


    def train(self, inputs, outputs, lr):
        self.sum = 0
        self.learning_rate = lr
        input = Matrix.fromArray(inputs)

        hidden_ip = self.weight_hi.multiply(input)
        hidden_ip.add(self.bias_ih)

        hidden_op = Matrix.map(hidden_ip, self.sigmoid)

        final_op = self.weight_oh.multiply(hidden_op)
        final_op.add(self.bias_oh)
        final_op = Matrix.map(final_op, self.scale) 

        for i in range(final_op.rows):
            for j in range(final_op.columns):
                self.sum += np.exp(final_op.matrix[i][j])
        
        final_op = Matrix.map(final_op, self.softmax)
        final_op.display()

        output = Matrix.fromArray(outputs)
        output = output.transpose()

        error = Matrix.subtract_stat(output, final_op)
        gradient = Matrix.map(final_op, self.dsoftmax)
        gradient = gradient.element_wise_multiply(error)
        gradient.scalar(self.learning_rate)

        self.bias_oh.add(gradient)
        hidden_op_t = hidden_op.transpose()
        weight_deltas = gradient.multiply(hidden_op_t)
        self.weight_oh.add(weight_deltas)

        who_t = self.weight_oh.transpose()
        error = error.transpose()
        hidden_error = who_t.multiply(error)

        hidden_gradient = Matrix.map(hidden_op, self.dsigmoid)
        hidden_gradient = hidden_gradient.element_wise_multiply(hidden_error)
        hidden_gradient.scalar(self.learning_rate)
        self.bias_ih.add(hidden_gradient)

        input_t = input.transpose()
        weight_deltas_ih = hidden_gradient.multiply(input_t)
        self.weight_hi.add(weight_deltas_ih)

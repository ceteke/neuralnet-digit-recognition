import numpy as np
from minst import load_mnist
import matplotlib.pyplot as plt
from scipy.special import expit
from pylab import *
from random import randint
import time
import getopt
import sys

class NeuralNetwork(object):
    def __init__(self,layers,learning_parameter):
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.learning_parameter = learning_parameter
        self.layers = layers
        self.errors = []

    def activation_function(self,z,deriv=False):
        if deriv:
            .5 * (1 + np.tanh(.5 * z)**2*0.5)
        return .5 * (1 + np.tanh(.5 * z))

    def feed_forward(self, X):
        y = np.array(X)
        for w in self.weights:
            y = self.activation_function(np.dot(w, y))
        return y

    def back_propagete(self, X, y,i):
        #Store each activation and vector values
        activation = X
        activations = [X]
        zs = []
        for weight in self.weights:
            z = np.dot(weight, activation)
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        error = activations[-1] - y
        squared_error = 0.0
        for e in error:
            squared_error += e**2.0
        squared_error /= 2.0
        self.errors.append(squared_error[0])
        #First calculate the delta of output
        deltas = [None] * len(self.weights)
        deltas[-1] = error * self.activation_function(zs[-1],deriv=True)

        for i in xrange(2, len(self.layers)):
            delta = np.dot(self.weights[-i+1].T, deltas[-i+1]) * self.activation_function(zs[-i], deriv=True)
            deltas[-i] = delta

        deltas = np.array(deltas)

        for i in range(0, len(activations)-1):
            dEdW = np.dot(deltas[i],activations[i].T)
            self.weights[i] = self.weights[i] - self.learning_parameter*dEdW

def train(NN):
    print "Reading training data..."
    images_training, training_labels = load_mnist('training')
    training_inputs = [np.reshape(x, (784, 1))/255.0 for x in images_training]
    print "Training NN..."
    for i in range(0,len(images_training)):
        image = training_inputs[i]
        desired_output = get_desired_output(training_labels[i][0])
        NN.back_propagete(image, desired_output, i)
    print "Squared Error after feeding forward first training data: ", NN.errors[0]
    print "Squared Error after feeding forward last training data: ", NN.errors[len(images_training)-1]

def test(NN, is_single):
    wrong_count = 0
    correct_count = 0
    print "Reading test data..."
    images_testing, testing_labels = load_mnist('testing')
    testing_inputs = [np.reshape(x, (784, 1))/255.0 for x in images_testing]
    if is_single:
        index = randint(0, len(images_testing))
        image = testing_inputs[index]
        label = testing_labels[index][0]
        output = NN.feed_forward(image)
        prediction = output.argmax()
        print "I predict:",prediction
        imshow(images_testing[index], cmap=cm.gray)
        show()
        return

    print "Running tests"
    for i in range(0,len(testing_inputs)):
        image = testing_inputs[i]
        label = testing_labels[i][0]
        output = NN.feed_forward(image)
        prediction = output.argmax()
        if prediction == label:
            correct_count += 1
        else:
            wrong_count += 1
    accuracy = float(correct_count) / float(wrong_count+correct_count)
    print "Accuracy:",accuracy

layers =[784,100,10]
learning_parameter = 0.2

def get_desired_output(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def handle_arguments(argv):
    optlist, args = getopt.getopt(argv, '', ['layers=', 'learning_parameter='])
    global layers
    global learning_parameter
    for opt in optlist:
        if '--layers' in opt:
            del layers[1]
            layer_sizes = [int(l) for l in opt[1].split(',')]
            layers[1:1] = layer_sizes
        elif '--learning_parameter' in opt:
            learning_parameter = float(opt[1])
        else:
            usage()

def usage():
    print "--layers '<hidden,layer,sizes,sperated,by,comma,no,space' --learning_parameter <parameter> "

handle_arguments(sys.argv[1:])
start_time = time.time()
NN = NeuralNetwork(layers,learning_parameter)
train(NN)
test(NN,False)
print("--- Time took: %s seconds ---" % (time.time() - start_time))

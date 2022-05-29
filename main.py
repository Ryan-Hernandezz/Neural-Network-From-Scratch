from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy
from sklearn.model_selection import train_test_split
import time

#datasets
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x / 255).astype('float32')
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

class NeuralNetwork():
    def __init__(self, sizes, epochs = 10, l_rate = 0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        
        self.parameters = self.initialization()
    
    #sigmoid function and its derivative
    def sigmoid(self, x, der = False):
        if der:
            return (numpy.exp(-x) / (1 + numpy.exp(-x))**2)
        return (1 / (1 + numpy.exp(-x)))
    
    #softmax function and its derivative
    def softmax(self, x, der = False):
        exps = numpy.exp(x - x.max())
        if der:
            return exps / numpy.sum(exps, axis=0) * (1 - exps / numpy.sum(exps, axis=0))
        return exps / numpy.sum(exps, axis=0)
    
    def initialization(self):
        input_layer = self.sizes[0]
        hidden_layer1 = self.sizes[1]
        hidden_layer2 = self.sizes[2]
        output_layer = self.sizes[3]
        #weights
        parameters = {
            'W1' :numpy.random.randn(hidden_layer1, input_layer) * numpy.sqrt(1. / hidden_layer1),
            'W2' :numpy.random.randn(hidden_layer2, hidden_layer1) * numpy.sqrt(1. / hidden_layer2),
            'W3' :numpy.random.randn(output_layer, hidden_layer2) * numpy.sqrt(1. / output_layer)
        }
        return parameters
    
    #apply the dot operation to each layer
    def move_foward(self, x_train):
        parameters = self.parameters
        
        parameters['A0'] = x_train
        parameters['Z1'] = numpy.dot(parameters['W1'], parameters['A0'])
        parameters['A1'] = self.sigmoid(parameters['Z1'])
        parameters['Z2'] = numpy.dot(parameters['W2'], parameters['A1'])
        parameters['A2'] = self.sigmoid(parameters['Z2'])
        parameters['Z3'] = numpy.dot(parameters['W3'], parameters['A2'])
        parameters['A3'] = self.softmax(parameters['Z3'])
        
        return parameters['A3']
    
    
    def move_back(self, y_train, output):
        parameters = self.parameters
        change_weights = {}
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(parameters['Z3'], derivative=True)
        change_weights['W3'] = numpy.outer(error, parameters['A2'])
        error = numpy.dot(parameters['W3'].T, error) * self.sigmoid(parameters['Z2'], derivative=True)
        change_weights['W2'] = numpy.outer(error, parameters['A1'])
        error = numpy.dot(parameters['W2'].T, error) * self.sigmoid(parameters['Z1'], derivative=True)
        change_weights['W1'] = numpy.outer(error, parameters['A0'])
        return change_weights
    
    
    def update_parameters(self, change_weights):
        for key, value in change_weights.items():
            self.parameters[key] -= self.l_rate * value
    
    
    def get_accuracy(self, x_val, y_val):
        predictions = []
        for x, y, in zip(x_val, y_val):
            output = self.move_foward(x)
            pre = numpy.argmax(output)
            predictions.append(pre == numpy.argmax(y))
        return numpy.mean(predictions)
    
    
    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.move_foward(x)
                changes_to_weights = self.move_back(y, output)
                self.update_parameters(changes_to_weights)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(iteration+1, time.time() - start_time, accuracy * 100))
            

dnn = NeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)

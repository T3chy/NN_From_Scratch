import sys
import numpy as np
import matplotlib.pyplot as plt
E = 2.71828182846
def create_data(points,classes):
    x = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number,points*(class_number+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*.05
        x[ix] = np.c_[r*np.sin(t*2.5),r*np.cos(t*2.5)]
        y[ix] = class_number
        return x,y
class Layer_Dense:
    def __init__(self,inputs,neurons):
        # init weights and biases
        self.weights = .01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros(shape=(1,neurons))
    def forward(self,inputs):
        #calc output values from inputs w/ weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs = np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probs
X, y = create_data(100,3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
activation1 = Activation_ReLU()
activation1.forward(dense1.output[:5])
print(activation1.output[:5])
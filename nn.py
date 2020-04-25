import sys
import numpy as np
import matplotlib.pyplot as plt
import math
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
    def __init__(self,inputs,neurons,weight_regularizer_l1=0,weight_regularizer_l2=0,bias_regularizer_l1=0,bias_regularizer_l2=0):
        # init weights and biases
        self.weights = .01 * np.random.randn(inputs,neurons)
        self.biases = np.zeros(shape=(1,neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    def forward(self,inputs):
        #calc output values from inputs w/ weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs # for backpropagation
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dvalues = np.dot(dvalues,self.weights.T)
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probs
    def backward(self,dvalues):
        self.dvalues = dvalues.copy()
class Loss:
    def regularization_loss(self,layer):
        regularization_loss = 0
        # L1 regularization- weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # L2 regularization- biases
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)
        # L1 regularization- weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # L2 regularization- biases
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)
        return regularization_loss
class Loss_categoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        print(self)
        samples = len(y_pred)
        y_pred = y_pred[range(samples),y_true]
        negative_log_likelihoods = -np.log(y_pred)
        data_loss = np.mean(negative_log_likelihoods)
        return data_loss
    def backward(self,dvalues,y_true):
        samples = dvalues.shape[0]
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples
class Optimizer_SGD:
    def __init__(self,learning_rate=1,decay=0.,momentum=0):
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1./ (1. + self.decay * self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        if self.momentum:
            weight_updates = (
                (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights)
            )
            layer.weight_momentums = weight_updates
            bias_updates = (
                (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases)
            )
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adagrad:
    def __init__(self,learning_rate=1,decay=0.,epsilon=1e-7):
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1./ (1. + self.decay * self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
class Optimizer_RMSprop:
    def __init__(self,learning_rate=1,decay=0.,epsilon=1e-7,rho=0.9):
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1./ (1. + self.decay * self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
class Optimizer_Adam:
    def __init__(self,learning_rate=.001,decay=0.,epsilon=1e-7,beta_1=0.9,beta_2=.999):
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1./ (1. + self.decay * self.iterations))
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentunms = self.beta_1 * layer.weight_cache + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_cache + (1-self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1- self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1- self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1
X, y = create_data(100,3)
print(X,y)
dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
activation2 = Activation_ReLU()
loss_function = Loss_categoricalCrossEntropy()
optimizer =Optimizer_SGD()
for epoch in range(1000000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.forward(activation2.output,y)
    predicitons = np.argmax(activation2.output,axis=1)
    accuracy = np.mean(predicitons==y)
    print("epoch:",epoch,'accuracy:',accuracy,'loss:',loss)
    # backward pass
    loss_function.backward(activation2.output,y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)
    # update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
X_test, y_test = create_data(100,3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.forward(activation2.output,y_test)
predictions = np.argmax(activation2.output,axis=1)
accuracy = np.mean(predictions==y_test)
print("validation: accuracy:",accuracy,'loss:',loss)
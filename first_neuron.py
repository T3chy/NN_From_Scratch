import sys
import numpy as np
import matplotlib
inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-.8]]
weights = [[.2,.8,-.5,1],[.5,-.91,.26,-.5],[-.26,-.27,.17,.87]]
biases = [2,3,.5]
layer_outputs = np.dot(inputs,np.array(weights).T) + biases
print(layer_outputs)
import numpy as np
from scipy.special import expit

# Activation function constants
sigmoid = "-=-=-=-=-"
tanh = "-=-=-=-=-=-"
ReLU = "-=-=-=-=-=-=-"
LeakyReLU = "-=-=-=-=-=-=-=-"

# Functions from the original layer class
def Sigmoid(x):
    return expit(x, dtype=np.float32)

def dSigmoid(y):
    return np.float32(y) * (np.float32(1) - np.float32(y))

def reLU(x):
    return np.maximum(0, x, dtype=np.float32)

def dreLU(x):
    return (x > 0).astype(np.float32)

def leakyReLU(x):
    return np.maximum(0.01 * x, x, dtype=np.float32)

def dleakyReLU(y):
    # This checking of y (output) or x (input) depends on implementation. 
    # Original used x logic but function arg is y. 
    # Provided implementation: 
    # y1 = np.float32(y >= 0)
    # y2 = np.float32(y < 0) * np.float32(0.01)
    # For LeakyReLU, dy/dx = 1 if x>0 else 0.01. 
    # If y is passed (and y=x for x>0, y=0.01x for x<0), then y>=0 implies x>=0. 
    # So checking y is fine.
    # Optimizing:
    return np.where(y >= 0, 1.0, 0.01).astype(np.float32)

def Tanh(z):
    return np.tanh(z, dtype=np.float32)

def dTanh(y):
    return np.float32(1) - np.float32(y) * np.float32(y)

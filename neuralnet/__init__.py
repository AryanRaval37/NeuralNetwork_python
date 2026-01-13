from .network import NeuralNetwork
from .matrix import Matrix
from .layers import Layer
from .activations import *

# Alias for backward compatibility
matrix = Matrix

__all__ = [
    "NeuralNetwork",
    "Matrix",
    "matrix",
    "Layer",
    "sigmoid",
    "tanh",
    "ReLU",
    "LeakyReLU",
]

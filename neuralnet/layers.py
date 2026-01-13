import numpy as np
import warnings
from .matrix import Matrix
from . import activations

# Module-level variable to control warnings, set by NeuralNetwork
supressWarnings = False

class Layer:
    def __init__(
        self, name=None, nodes=None, units=None, special=None, activation=None
    ):
        if nodes is None and units is None:
            assert (
                False
            ), "\n\nThe number of nodes/units in the layer are not specified.\n"
        if nodes is not None:
            self.nodes = nodes
        if units is not None:
            self.nodes = units
        if self.nodes > 6400 or self.nodes is None:
            assert False, "\n\nThe number of nodes is not valid.\n"

        self.name = name

        if activation is None:
            if not supressWarnings:
                warnings.warn(
                    f"The activation function for the layer {self.name} is not given.\nUsing sigmoid activation instead."
                )
            activation = activations.sigmoid

        if activation in [activations.sigmoid, activations.ReLU, activations.LeakyReLU, activations.tanh]:
            self.activationVar = activation
            if activation == activations.sigmoid:
                self.activation = activations.Sigmoid
                self.dactivation = activations.dSigmoid
            elif activation == activations.ReLU:
                self.activation = activations.reLU
                self.dactivation = activations.dreLU
            elif activation == activations.LeakyReLU:
                self.activation = activations.leakyReLU
                self.dactivation = activations.dleakyReLU
            elif activation == activations.tanh:
                self.activation = activations.Tanh
                self.dactivation = activations.dTanh

        else:
            assert (
                False
            ), f"\n\nInvalid activation function given.\nReceived {activation}"

        if special == "InPuT_0":
            self.type = "INPUT"
        elif special == "OuTpUt_last":
            self.type = "OUTPUT"
        else:
            self.type = None

        self.weights = None
        self.bias = None
        self.key = None

    # inbuilt method to print the layer
    def __str__(self):
        if self.name is None:
            print(
                f"Unnamed Layer. Key : {self.key if (self.key is not None) else 'none'}"
            )
        else:
            print(
                f"Name : {self.name}\nKey : {self.key if (self.key is not None) else 'none'}"
            )
        print(f"Nodes : {self.nodes}")
        if self.type == "INPUT":
            print("\nThis is the input layer hold only temperary inputs.")
        if self.weights is None:
            print(
                "The layer is not yet added to the network. Compile the network to see the weights."
            )
        else:
            print("\nWeights : ")
            print(self.weights)
            print("Bias : ")
            print(self.bias)
        return ""

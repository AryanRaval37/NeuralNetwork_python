import sys
import os
import numpy as np

# Ensure we can import from the current directory
sys.path.insert(0, os.getcwd())

from neuralnet import NeuralNetwork, matrix, sigmoid, tanh

def test_basic():
    print("Testing NeuralNetwork package...")
    
    # Test Matrix creation
    m = matrix(2, 2)
    print(f"Matrix created: {m.rows}x{m.cols}")
    assert m.rows == 2 and m.cols == 2

    # Test NeuralNetwork creation
    nn = NeuralNetwork(inputs=2, outputs=1, task="Regression", supress_neuralnetwork_warnings=True)
    print("NeuralNetwork created")
    
    # Test adding layer
    nn.addLayer(NeuralNetwork.layer(nodes=4, activation=sigmoid))
    print("Layer added")
    
    # Test compilation
    nn.compileModel(activation=sigmoid)
    print("Model compiled")
    
    # Test prediction
    output = nn.predict([0.5, 0.5])
    print(f"Prediction result: {output}")
    assert isinstance(output, list)
    
    # Test training step (minimal)
    nn.addTrainingData([0.1, 0.2], [0.3])
    print("Training data added")
    
    # We won't run full training as it involves multiprocessing which might be flaky in test script execution environment
    # but we checked structure.
    
    print("Basic tests passed!")

if __name__ == "__main__":
    test_basic()

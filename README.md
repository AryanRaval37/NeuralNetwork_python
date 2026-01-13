# NeuralNetwork Python Library

This is a simple Neural Network library built from scratch in Python.

## Refactoring to `neuralnet` Package

The library has been refactored from a single `NeuralNetwork.py` file into a maintainable Python package named `neuralnet`.

### Process Followed

1.  **Directory Structure**: Created a `neuralnet` directory to house the package.
2.  **Modularization**: Split the code into logical modules:
    -   `neuralnet/activations.py`: Contains activation functions and constants (`sigmoid`, `tanh`, `ReLU`, `LeakyReLU`).
    -   `neuralnet/matrix.py`: Contains the `Matrix` class (formerly `matrix`).
    -   `neuralnet/layers.py`: Contains the `Layer` class (formerly `NeuralNetwork.layer`).
    -   `neuralnet/network.py`: Contains the main `NeuralNetwork` class.
3.  **Encapsulation**:
    -   Moved `Matrix` class to its own file.
    -   Moved `Layer` class definition out of `NeuralNetwork` class.
    -   Updated `NeuralNetwork` to import `Layer` and `Matrix`.
4.  **Backward Compatibility**:
    -   In `neuralnet/__init__.py`, aliases were created to ensuring `from neuralnet import *` works similarly to the old import.
    -   `matrix` is aliased to `Matrix`.
    -   `NeuralNetwork.layer` is preserved by adding `layer = Layer` inside the `NeuralNetwork` class.
5.  **Installation**: Added `setup.py` to allow installation via pip.

### Installation

To install the package in editable mode (so changes are reflected immediately):

```bash
pip install -e .
```

### Usage

**Old Way:**
```python
from NeuralNetwork import *
```

**New Way:**
```python
from neuralnet import *
```

The API remains largely the same.

### Files

-   `neuralnet/`: Package source code.
-   `setup.py`: Installation script.
-   `tests/`: Test scripts.
-   `examples/`: Example usages.
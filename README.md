# NeuralNetwork Python Library

A powerful, highly customizable, and easy-to-use Neural Network library built from scratch in Python. Whether you are building a simple regression model or a complex doodle classifier, this library provides the tools you need with modern features like Mini-Batch Gradient Descent and real-time training visualization.

---

## 🚀 Installation

1.  **Clone or Download** the repository.
2.  Navigate to the root directory.
3.  Install via pip:

```bash
pip install .
```

To install in editable mode (so changes to the code reflect immediately):

```bash
pip install -e .
```

---

## 🔰 Quick Start

Here is a simple example to solve the **XOR** problem (Classification/Regression mixed task).

```python
from neuralnet import NeuralNetwork, Layer, activations

# 1. Initialize
# Inputs: 2, Outputs: 1
nn = NeuralNetwork(inputs=2, outputs=1, task="Regression") 

# 2. Build Network
nn.addLayer(Layer(nodes=4, activation=activations.tanh, name="Hidden"))
nn.addLayer(Layer(nodes=1, activation=activations.sigmoid, name="Output"))

# 3. Compile
nn.compileModel(activation=activations.sigmoid)

# 4. Data
training_data = [
    {"input": [0, 0], "target": [0]},
    {"input": [0, 1], "target": [1]},
    {"input": [1, 0], "target": [1]},
    {"input": [1, 1], "target": [0]},
]
nn.addData(training_data, threshold=1.0) 

# 5. Train
nn.train(epochs=5000, batch_size=4, log_freq=100) 

# 6. Predict
print(nn.predict([1, 0]))
```

---

## 📚 Detailed Documentation

### 1. Initialization

The entry point is the `NeuralNetwork` class.

```python
from neuralnet import NeuralNetwork

nn = NeuralNetwork(inputs, outputs=None, labels=None, task="Regression", supress_neuralnetwork_warnings=False)
```

**Arguments:**
-   **`inputs`** *(int)*: Number of input features (neurons).
-   **`outputs`** *(int, optional)*: Number of output neurons. **Required if `task="Regression"`**.
-   **`labels`** *(list of str, optional)*: List of class names (e.g., `["cat", "dog"]`). **Required if `task="Classification"`**.
-   **`task`** *(str)*: Defines the problem type. Options: `"Regression"`, `"Classification"`.
-   **`supress_neuralnetwork_warnings`** *(bool)*: If `True`, silences non-critical warnings.

---

### 2. Building the Network

Add layers to your network sequentially.

#### `addLayer(layer)`

Adds a new layer to the stack.

```python
nn.addLayer(Layer(nodes, activation, name=None))
```

-   **`nodes`** *(int)*: Number of neurons in this layer.
-   **`activation`** *(function)*: Activation function from `neuralnet.activations`.
    -   `activations.sigmoid`
    -   `activations.tanh`
    -   `activations.relu`
    -   `activations.leakyRelu` (Leaky ReLU)
    -   `activations.softmax` (Typically for output layer in classification)
-   **`name`** *(str, optional)*: A descriptive name for the layer.

#### `compileModel(activation=activations.sigmoid)`

Finalizes the model structure. **Must be called before adding data or training.**

-   **`activation`**: default activation function if a layer doesn't specify one.

#### `setLearningRate(lr)`

Sets the learning rate for the optimizer.

-   **`lr`** *(float)*: A value between 0 and ~2.5. Default is usually `0.1`.

---

### 3. Data Management

The library supports two main data formats depending on your task.

#### `addData(data, threshold=0.8)`

Adds a list of data points and automatically splits them into training and testing sets.

-   **`data`** *(list of dict)*: Your dataset.
-   **`threshold`** *(float)*: Percentage of data to use for training (0.0 to 1.0). E.g., `0.8` means 80% Training, 20% Testing.

**Format for Regression:**
```python
data = [
    {"input": [0.1, 0.2], "target": [0.5]}, 
    # ...
]
```

**Format for Classification:**
```python
data = [
    {"input": [0.1, 0.2], "label": "cat"}, 
    # ...
]
```

#### `addTrainingData(input, output)` & `addTestingData(input, output)`
Manually add single data points if you don't want to use `addData`.

---

### 4. Training

#### `train(...)`

Starts the training process. This runs in a separate process to allow for real-time visualization.

```python
nn.train(
    epochs, 
    batch_size=32, 
    log_freq=10, 
    debug=True, 
    whileTraining=None
)
```

-   **`epochs`** *(int)*: Total number of passes through the **entire** dataset.
-   **`batch_size`** *(int)*: Number of samples processed before updating weights. 
    -   Higher values (e.g., 64, 128, 4000) are faster (matrix operations) but use more RAM.
    -   Lower values (e.g., 1, 4) update weights more frequently but are slower.
-   **`log_freq`** *(int)*: **Plotting Frequency**. 
    -   Controls how often (in **batches**) the loss graph updates.
    -   **1**: Update every batch (Real-time, smooth).
    -   **100**: Update every 100 batches (Less overhead).
-   **`debug`** *(bool)*: If `True`, launches a pop-up window plotting the Loss vs Steps.
-   **`whileTraining`** *(function)*: A callback function `func(epoch, loss)` called after every epoch. Useful for updating a progress bar or UI.

---

### 5. Prediction & Evaluation

#### `predict(input_array)`
*(Regression)*
Forward passes the input through the network and returns the raw output vector.

```python
output = nn.predict([0.5, 0.1])
# Returns: [0.892]
```

#### `classify(input_array, applySoftmax=True)`
*(Classification)*
returns the predicted class/label.

```python
result = nn.classify([0.5, 0.1])
# Returns: [{'class': 'cat', 'probability': 0.95}, ...]
```

#### `runTest(applySoftmax=True, parallel=5)`
*(Classification only)*
Runs the model against the testing dataset added via `addData` or `addTestingData` and returns accuracy metrics.

```python
stats = nn.runTest()
print(f"Accuracy: {stats['accuracy']}%")
```

---

### 6. Saving and Loading

You can save your trained model to a JSON file and load it later.

#### `save(filename, infoName=None, moreInfo={})`

```python
nn.save("my_model.json")
```
-   **`filename`**: Path to save the file.
-   **`infoName`**: Optional name metadata for the model.
-   **`moreInfo`**: Dictionary of extra metadata to save.

#### `load(filename)`

Loads weights into an **existing, initialized** network. Structure must match.

```python
nn.load("my_model.json")
```

#### `NeuralNetwork.fromFile(filename)`

Static method. Creates a **new** `NeuralNetwork` instance completely from the file.

```python
nn = NeuralNetwork.fromFile("my_model.json")
```

---

### 7. Custom Layers & Matrix

You can also use the low-level `Matrix` class for math operations.

```python
from neuralnet import Matrix
m1 = Matrix(2, 2)
m1.randomize()
m2 = Matrix.transpose(m1)
```

---

### Key Concepts

-   **Mini-Batch Gradient Descent**: The library processes data in chunks (`batch_size`). This is much faster than Stochastic Gradient Descent (one by one) because it uses optimized Matrix operations.
-   **Gradient Normalization**: Gradients are automatically divided by `batch_size` to ensure the `learning_rate` behaves consistently regardless of how large your batch is.
-   **Unified Plotting**: The `log_freq` parameter allows you to control visualization. It plots the **average loss** over the frequency interval, ensuring smooth, professional-looking graphs.
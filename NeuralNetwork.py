from Matrix import Matrix as matrix
import math


class NeuralNetwork:
    def __init__(self, obj):
        if "Inputs" in obj.keys():
            self.inputNodes = obj["Inputs"]
        elif "inputs" in obj.keys():
            self.inputNodes = obj["inputs"]
        if "Outputs" in obj.keys():
            self.outpuNodes = obj["Outputs"]
        elif "outputs" in obj.keys():
            self.outputNodes = obj["outputs"]
        assert (
            self.inputNodes <= 6400 or self.inputNodes is None
        ), "\n\nInvalid number of input nodes."
        assert (
            self.outputNodes <= 6400 or self.outputNodes is None
        ), "\n\nInvalid number of output nodes."
        self.layers = []
        self.compiled = False
        self.layers.append(
            self.layer(
                {
                    "name": "Input_layer",
                    "nodes": self.inputNodes,
                    "SecretType_409612341024": "INPUT",
                }
            )
        )
        self.learning_rate = 0.1

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def __dsigmoid(y):
        return y * (1 - y)

    def __predictLayer(self, index_l2, results_l1):
        inputs = results_l1
        hidden = matrix.multiply(self.layers[index_l2].weights, inputs)
        hidden.add(self.layers[index_l2].bias)
        outputs = matrix.map(hidden, self.sigmoid)
        return outputs

    def predict(self, input_array):
        # predict function works. However it is giving numbers very close to one when the number
        # of layers are increased. Guessed Reaseon. There are no negative numbers here.
        # every layer, the bias gets added (which is random)
        # multiplication hardly recuded the numbers as compared the addition of the bias.
        # is this supposed to happen??
        # will this give an accurate result when trained???
        assert isinstance(
            input_array, list
        ), f"\nThe input to predict funtion is not a list.\nReceived {type(input_array)}.\n"
        assert (
            len(input_array) == self.inputNodes
        ), "\n\nThe number of inputs provided to predict are not matching\nto the number of inputNodes mentioned before."
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet.\n Compile the model to predict outputs.\n"
        self.layers[0].inputList = input_array
        self.layers[0].inputMatrix = matrix.toMatrix(input_array, "InputList")
        print(self.layers[0].inputMatrix)
        i = 2
        previousPrediction = self.__predictLayer(1, self.layers[0].inputMatrix)
        print(previousPrediction)
        while i <= len(self.layers) - 1:
            print(i)
            print(previousPrediction)
            previousPrediction = self.__predictLayer(i, previousPrediction)
            i += 1
        print(previousPrediction)
        prediction = previousPrediction.toList()
        return prediction

    def __connect(self, index_l1, index_l2):
        nodes_l1 = self.layers[index_l1].nodes
        nodes_l2 = self.layers[index_l2].nodes
        self.layers[index_l2].weights = matrix(nodes_l2, nodes_l1)
        self.layers[index_l2].weights.randomize()
        self.layers[index_l2].bias = matrix(nodes_l2, 1)
        self.layers[index_l2].bias.randomize()

    def addLayer(self, Layer):
        assert isinstance(
            Layer, self.layer
        ), f"\nInvalid argument to addLayer.\nExpected layer but reveived {type(Layer)}"
        assert (
            self.compiled == False
        ), "\nThe model is already compiled. Another layer cannot be added to the network."
        if Layer.name is None:
            Layer.name = f"unnamed_layer{len(self.layers)}"
        Layer.key = len(self.layers)
        self.layers.append(Layer)

    def compileModel(self):
        self.layers.append(
            self.layer(
                {
                    "name": "Output_layer",
                    "nodes": self.outputNodes,
                    "SecretType_409612341024": "OUTPUT",
                }
            )
        )
        for i in range(len(self.layers) - 1):
            self.__connect(i, i + 1)
        self.compiled = True

    class layer:
        def __init__(self, obj):
            if "name" in obj.keys():
                self.name = obj["name"]
            elif "Name" in obj.keys():
                self.name = obj["Name"]
            else:
                self.name = None

            if "Nodes" in obj.keys():
                self.nodes = obj["Nodes"]
            elif "nodes" in obj.keys():
                self.nodes = obj["nodes"]
            else:
                assert False, "\n\nError forming layer. Number of nodes is not given.\n"

            if self.nodes > 6400 or self.nodes is None:
                assert False, "\n\nThe number of nodes is not valid.\n"

            if "SecretType_409612341024" in obj.keys():
                self.type = obj["SecretType_409612341024"]
                if self.type == "INPUT":
                    self.inputList = None
                    self.inputMatrix = None
                if self.type == "OUTPUT":
                    self.outputList = None
                    self.outputMatrix = None
            else:
                self.type = None

            self.weights = None
            self.bias = None
            self.key = None

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

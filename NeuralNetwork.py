from Matrix import Matrix as matrix
import math


class NeuralNetwork:
    def __init__(self, obj):
        # checking if there are any inputs or outputs as keys in obj
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
        # layers in the model
        self.layers = []
        # boolean to check if the model is compiled.
        self.compiled = False
        # adding the input layer to the network the moment it is created.
        self.layers.append(
            self.layer(
                {
                    "name": "Input_layer",
                    "nodes": self.inputNodes,
                    "SecretType_409612341024": "INPUT",
                }
            )
        )
        # learning rate of the network
        self.learning_rate = 0.1
        # boolean to check if the model is training
        self.isTraining = False

    # These functions cannot be private as they cannot be called by the matrix library.
    # the Activation function of the network
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # derivative of Sigmoid function       |*_*| CALCULUS |*_*|
    def dsigmoid(y):
        return y * (1 - y)

    # continue creating train function...
    def train(self):
        assert (
            self.isTraining == False
        ), "\nThe model is already training.\nTrain was called again."
        assert (
            self.compiled
        ), "\nThe model is not compiled. Training can only be done after the model is compiled."

    # private function which calculates the generated between two layers.
    def __predictLayer(self, index_l2, results_l1):
        # inputs of the previous layer
        inputs = results_l1
        # matrix product of the weights and the inputs
        outputs = matrix.multiply(self.layers[index_l2].weights, inputs)
        # adding the layers bias
        outputs.add(self.layers[index_l2].bias)
        # mapping it to a range between 0 and 1 by the sigmoid function
        outputs.map(self.sigmoid)
        # returning the outputs
        return outputs

    # public function the predict the outputs for given inputs
    def predict(self, input_array):

        # predict function works. However it is giving numbers very close to one when the number
        # of layers are increased. Guessed Reaseon. There are no negative numbers here.
        # every layer, the bias gets added (which is random)
        # multiplication hardly recuded the numbers as compared the addition of the bias.
        # is this supposed to happen??
        # will this give an accurate result when trained???

        # The error cause given above is WRONG...  ✌️ ✌️ ✌️
        # Error : The weights of the network should be initiallized from -1 to 1
        #         and not 0 to 1.

        # This fixed the problem. Remove This in the next commit.

        assert isinstance(
            input_array, list
        ), f"\nThe input to predict funtion is not a list.\nReceived {type(input_array)}.\n"
        assert (
            len(input_array) == self.inputNodes
        ), "\n\nThe number of inputs provided to predict are not matching\nto the number of inputNodes mentioned before."
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet.\n Compile the model to predict outputs.\n"
        assert (
            self.isTraining == False
        ), "\n\nThe model is training it cannot predict results."

        # adding the inputs to the input layer.
        self.layers[0].inputList = input_array
        # converting it to a matrix
        self.layers[0].inputMatrix = matrix.toMatrix(input_array, "InputList")

        # just for monitoring progress do remove later
        print(self.layers[0].inputMatrix)
        # producing outputs for the inputs with the first layer.
        previousPrediction = self.__predictLayer(1, self.layers[0].inputMatrix)
        print(previousPrediction)

        # innitially i has to be two as the outputs of the first layer with the 0 layer
        # (input layer) are alrealy done.
        i = 2
        while i <= len(self.layers) - 1:
            # this is also for monitoring progress remove this later.
            print(i)
            print(previousPrediction)
            previousPrediction = self.__predictLayer(i, previousPrediction)
            i += 1

        # remove later
        print(previousPrediction)
        # converting the predictions to a list and then returning the list
        prediction = previousPrediction.toList()
        return prediction

    # private function to connect two layers:
    # Connecting = creating matrices of suitable length and initiallzing randomly
    def __connect(self, index_l1, index_l2):
        nodes_l1 = self.layers[index_l1].nodes
        nodes_l2 = self.layers[index_l2].nodes
        self.layers[index_l2].weights = matrix(nodes_l2, nodes_l1)
        self.layers[index_l2].weights.randomize()
        self.layers[index_l2].bias = matrix(nodes_l2, 1)
        self.layers[index_l2].bias.randomize()

    # function too add an extra layer to the network
    def addLayer(self, Layer):
        assert isinstance(
            Layer, self.layer
        ), f"\nInvalid argument to addLayer.\nExpected layer but reveived {type(Layer)}"
        assert (
            self.compiled == False
        ), "\nThe model is already compiled. Another layer cannot be added to the network."
        assert (
            self.isTraining == False
        ), "\n\nThe model is training it cannot predict results."
        if Layer.name is None:
            Layer.name = f"unnamed_layer{len(self.layers)}"
        Layer.key = len(self.layers)
        self.layers.append(Layer)

    # function to add output layer at the end and finalize the model
    # Connecting the layers = Initiallizing random weights
    def compileModel(self):
        assert (
            self.isTraining == False
        ), "\n\nThe model is training it cannot predict results."
        assert (
            self.compiled == False
        ), "\n\nThe model is already compiled.\n It cannot be recompiled."
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

    # layer class (inner class of the NeuralNetwork class)
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

        # inbuilt method to print the layer
        # print(layer) will give these results
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

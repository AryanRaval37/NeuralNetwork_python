import numbers
from Matrix import Matrix as matrix
import math
import random
import concurrent.futures
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Error : on Plotting the Epochs.
# Not showing correct number of epochs in plot
# Guessed reason :
#   1) The Matplotlib plot starts after a while after the actual whileTraining function begins.
#   2) The training case here is very easy. Train puts data in the queue very fast.
#      Queue.get() extracts the data one at a time so the queue instead of having only one element starts filling up.
#      Try queue.get_nowait() and see if you get exception of queue empty to test error logic.
# Therefore here check error and display the correct number of Epochs. Change plot styles.
# Increase animation frame rate to max possible. Decrease number of times losses are plotted.
# if not fixed make a system to remove previous queue element if new element needs to come.
# or make a system of queues sharing only one element of data.
# Fix error till next Commit and then merge with main branch.


class NeuralNetwork:
    def __init__(self, inputs, outputs):
        # checking if there are any inputs or outputs as keys in obj
        self.inputNodes = inputs
        self.outputNodes = outputs
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
        # training data in the network
        self.data = []
        # adding the input layer to the network the moment it is created.
        self.layers.append(
            self.layer(name="Input_Layer", nodes=self.inputNodes, special="InPuT_0")
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
    def dsigmoid(self, y):
        return y * (1 - y)

    def addData(self, input_array, target_array):
        assert self.inputNodes == len(
            input_array
        ), "\n\nThe inputs provided to the addData function do match number of inputs mentioned earlier."
        assert self.outputNodes == len(
            target_array
        ), "\n\nThe targets provided to the addData function do match number of outputs mentioned earlier."
        self.data.append({"input": input_array, "target": target_array})

    # private function which calculates the generated between two layers.
    def predictLayer(self, index_l2, results_l1):
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

    def mapLR(self, x):
        return x * self.learning_rate

    # function to change the learning rate of the network.
    def setLearningRate(self, lr):
        assert isinstance(
            lr, numbers.Number
        ), "\nThe learning rate given is not a number."
        assert lr > 0 and lr <= 2.5, "\nInvalid learning rate given."
        self.learning_rate = lr

    def mapLoss(self, x):
        return x * x / 2

    @staticmethod
    def myLossPlotter(queue, endQueue):
        anim = None
        data = []
        fig, ax = plt.subplots()

        def animate(i):
            if not queue.empty():
                queueData = queue.get()
                data.append(queueData[0])
                ax.clear()
                ax.plot(data)
                # ax.set_title("Training Process")
            if not endQueue.empty():
                shouldPlot = endQueue.get()
                if not shouldPlot:
                    anim.event_source.stop()

        anim = FuncAnimation(fig, animate, interval=5)
        # plt.style.use("fivethirtyeight")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss / Cost")
        plt.show()

    @staticmethod
    def trainNotToBeUsed(nn, queue, epochs, plot_interval, lossQueue):
        epochLosses = []
        k = 1
        queue1 = Queue()
        endQueue = Queue()
        plotingProcess = Process(
            target=NeuralNetwork.myLossPlotter, args=[queue1, endQueue]
        )
        plotingProcess.daemon = False
        plotingProcess.start()
        for epochCounter in range(epochs):
            random.shuffle(nn.data)
            # training an epoch
            epochLosses = []
            for i in range(len(nn.data)):
                input_array = nn.data[i]["input"]
                target_array = nn.data[i]["target"]
                # adding the inputs to the input layer.
                nn.layers[0].inputList = input_array
                # converting it to a matrix
                nn.layers[0].inputMatrix = matrix.toMatrix(input_array, "InputList")

                # producing outputs for the inputs with the first layer.
                previousPrediction = nn.predictLayer(1, nn.layers[0].inputMatrix)

                # innitially i has to be two as the outputs of the first layer with the 0 layer
                # (input layer) are alrealy done.
                i = 2
                layerPredictions = []
                layerPredictions.append(nn.layers[0].inputMatrix)
                layerPredictions.append(previousPrediction)
                while i <= len(nn.layers) - 1:
                    previousPrediction = nn.predictLayer(i, previousPrediction)
                    layerPredictions.append(previousPrediction)
                    i += 1
                outputs = previousPrediction

                # converting target list to matrix.
                targets = matrix.toMatrix(target_array)

                # ERROR = DESIRED - GUESS
                # Therefore first calculating guess and subtracting from target
                output_errors = matrix.subtract(targets, outputs)

                costMatrix = matrix.map_static(output_errors, nn.mapLoss)
                loss = costMatrix.mean()
                epochLosses.append(loss)

                Errors = output_errors
                i = len(nn.layers) - 1
                while i >= 1:
                    layer2 = nn.layers[i]

                    # calculaing gradients between layer i and i-1
                    gradients = matrix.map_static(layerPredictions[i], nn.dsigmoid)
                    gradients.simpleMultiply(Errors)
                    gradients.map(nn.mapLR)

                    # calculating change in weights
                    l1_predictions_transposed = matrix.transpose(
                        layerPredictions[i - 1]
                    )
                    weights_l2l1_deltas = matrix.multiply(
                        gradients, l1_predictions_transposed
                    )

                    # changing the weights of layer i
                    layer2.weights.add(weights_l2l1_deltas)
                    layer2.bias.add(gradients)

                    # calculating the errors for the next layer for next iteration in loop
                    weights_l2l1_transposed = matrix.transpose(layer2.weights)
                    Errors = matrix.multiply(weights_l2l1_transposed, Errors)

                    # reassigning layer2 to the actualy layers
                    nn.layers[i] = layer2
                    i -= 1
                # End of Backpropogation
            # End of going through all data (End of an EPOCH)
            sum = 0
            for i in range(len(epochLosses)):
                sum += epochLosses[i]
            meanLoss = sum / len(epochLosses)
            if epochCounter + 1 >= plot_interval * k:
                # checking if it is time to plot
                queue1.put([meanLoss, epochCounter + 1])
                k += 1
            # passing this to whileDoing function from main function call
            lossQueue.put([epochCounter + 1, meanLoss])
        # End of going through all the epochs (Training complete)
        changedWeights = [0]
        for i in range(1, len(nn.layers)):
            changedWeights.append(nn.layers[i])
        queue.put(changedWeights)
        lossQueue.put(False)
        endQueue.put(False)

    def train(self, whileTraining, onComplete, epochs=1, plotInterval=10):
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet.\n Compile the model to train..\n"
        assert self.isTraining == False, "\n\nThe model is already training."

        self.isTraining = True
        queue = Queue()
        lossQueue = Queue()
        trainingProcess = Process(
            target=self.__class__.trainNotToBeUsed,
            args=[self, queue, epochs, plotInterval, lossQueue],
        )
        trainingProcess.daemon = False
        trainingProcess.start()
        while trainingProcess.is_alive():
            EpochInfo = lossQueue.get()
            if not EpochInfo:
                break
            whileTraining(EpochInfo[0], EpochInfo[1])
        updatedLayers = queue.get_nowait()
        for i in range(1, len(self.layers)):
            self.layers[i] = updatedLayers[i]
        self.isTraining = False
        print("Done Training...")
        onComplete()

    # Cost / Loss function : ( Do Implement )
    #   C = 1/2 * (Guess - Desired)^2

    def trainData(self, input_array, target_array):
        assert isinstance(
            input_array, list
        ), f"\nThe input to predict funtion is not a list.\nReceived {type(input_array)}.\n"
        assert (
            len(input_array) == self.inputNodes
        ), "\n\nThe number of inputs provided to train are not matching\nto the number of inputNodes mentioned before."
        assert (
            len(target_array) == self.outputNodes
        ), "\n\nThe number of targets provided to the train function are not matching\nto the number of outputs mentioned before."
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet.\n Compile the model to train..\n"
        assert self.isTraining == False, "\n\nThe model is already training."

        self.isTraining = True

        # adding the inputs to the input layer.
        self.layers[0].inputList = input_array
        # converting it to a matrix
        self.layers[0].inputMatrix = matrix.toMatrix(input_array, "InputList")

        # producing outputs for the inputs with the first layer.
        previousPrediction = self.predictLayer(1, self.layers[0].inputMatrix)

        # innitially i has to be two as the outputs of the first layer with the 0 layer
        # (input layer) are alrealy done.
        i = 2
        layerPredictions = []
        layerPredictions.append(self.layers[0].inputMatrix)
        layerPredictions.append(previousPrediction)
        while i <= len(self.layers) - 1:
            previousPrediction = self.predictLayer(i, previousPrediction)
            layerPredictions.append(previousPrediction)
            i += 1
        outputs = previousPrediction

        # converting target list to matrix.
        targets = matrix.toMatrix(target_array)

        # ERROR = DESIRED - GUESS
        # Therefore first calculating guess and subtracting from target
        output_errors = matrix.subtract(targets, outputs)
        Errors = output_errors
        i = len(self.layers) - 1
        while i >= 1:
            layer2 = self.layers[i]

            # calculaing gradients between layer i and i-1
            gradients = matrix.map_static(layerPredictions[i], self.dsigmoid)
            gradients.simpleMultiply(Errors)
            gradients.map(self.mapLR)

            # calculating change in weights
            l1_predictions_transposed = matrix.transpose(layerPredictions[i - 1])
            weights_l2l1_deltas = matrix.multiply(gradients, l1_predictions_transposed)

            # changing the weights of layer i
            layer2.weights.add(weights_l2l1_deltas)
            layer2.bias.add(gradients)

            # calculating the errors for the next layer for next iteration in loop
            weights_l2l1_transposed = matrix.transpose(layer2.weights)
            Errors = matrix.multiply(weights_l2l1_transposed, Errors)

            # reassigning layer2 to the actualy layers
            self.layers[i] = layer2
            i -= 1
        self.isTraining = False

    # public function the predict the outputs for given inputs
    def predict_Async(self, input_array, onComplete):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.predict, input_array)]
            for f in concurrent.futures.as_completed(results):
                onComplete(f.result())

    def predict(self, input_array):
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

        # producing outputs for the inputs with the first layer.
        previousPrediction = self.predictLayer(1, self.layers[0].inputMatrix)

        # innitially i has to be two as the outputs of the first layer with the 0 layer
        # (input layer) are alrealy done.
        i = 2
        while i <= len(self.layers) - 1:
            previousPrediction = self.predictLayer(i, previousPrediction)
            i += 1
        # converting the predictions to a list and then returning the list
        prediction = previousPrediction.toList()
        self.PredictTest = True
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
                name="Output_Layer", nodes=self.outputNodes, special="OuTpUt_last"
            )
        )
        for i in range(len(self.layers) - 1):
            self.__connect(i, i + 1)
        self.compiled = True

    # layer class (inner class of the NeuralNetwork class)
    class layer:
        def __init__(self, name=None, nodes=None, units=None, special=None):
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

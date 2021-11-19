import numbers
import random
import concurrent.futures
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import warnings
import numpy as np
import copy

# NOW THE LIBRARY IS POWERED BY NUMPY
# atleast 25 times faster than previous version.
# numpy support in matrices.
# removed extra matrix file.

# DO's:
# take execution to GPU.
# implement more numpy as smallest of calculations and arrays.
# use mpmath for more precision while calculating numbers and matrices.
# Test more on Regression and remove small bugs.

# DONT's:
# Do not use mpmath.matrix directly.
# Will suffer slower execution than the old, slow looping matrix library.


class NeuralNetwork:
    def __init__(self, inputs, outputs=None, labels=None, task="Regression"):

        self.inputNodes = inputs
        assert task in [
            "Regression",
            "regression",
            "prediction",
            "Prediction" "classification",
            "Classification",
        ], "\n\nInvalid task provided. Cannot form network.\n"

        if task in ["Regression", "regression", "prediction", "Prediction"]:
            self.task = "Regression"
        elif task in ["Classification", "classification"]:
            self.task = "Classification"
        else:
            assert False, "\n\nInvalid task provided. Cannot form network.\n"

        # checking if at least one of them is valid
        assert outputs is not None or labels is not None, (
            "\n\nThe total number of labels are not provided to the network.\nError forming the network."
            if self.task == "Classification"
            else f"\n\nThe total number of outputs are not provided to the network.\nError forming the network."
        )

        # checking if both of them are given. Only one should be given.
        assert not (outputs is not None and labels is not None), (
            "\n\nBoth and outputs and labels were provided to the network.\nOnly "
            + ("labels " if self.task == "Classification" else "outputs ")
            + "was expected.\n Error forming the network."
        )

        # checking if given label/output match with the respective tasks
        if outputs is None and self.task == "Regression":
            assert (
                False
            ), "\nFor task Regression outputs should be given and not labels.\nError forming the network."
        elif labels is None and self.task == "Classification":
            assert (
                False
            ), "\nFor task Classification labels should be given and not outputs.\nError forming the network."

        if outputs is not None:
            self.outputNodes = outputs
        else:
            self.outputNodes = labels
            self.labels = []

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
        # testing data in the network
        self.testingData = []
        # adding the input layer to the network the moment it is created.
        self.layers.append(
            self.layer(name="Input_Layer", nodes=self.inputNodes, special="InPuT_0")
        )

        # learning rate of the network
        self.learning_rate = 0.1
        # boolean to check if the model is training
        self.isTraining = False
        warnings.simplefilter("default")

    # These functions cannot be private as they cannot be called by the matrix library.
    # the Activation function of the network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of Sigmoid function       |*_*| CALCULUS |*_*|
    def dsigmoid(self, y):
        return y * (1 - y)

    # function to add training data to the network
    def addData(self, input_array, output):
        if self.task == "Regression":
            assert isinstance(
                output, list
            ), f"\n\nThe output given to addData for task Regression is a {type(output)}.\nExpected a list."
            assert isinstance(
                input_array, list
            ), f"\n\nThe inputs provided to the addData function are of type {type(input_array)}.\nExpected list."
            assert self.inputNodes == len(
                input_array
            ), "\n\nThe inputs provided to the addData function do not match number of inputs mentioned earlier."
            assert self.outputNodes == len(
                output
            ), "\n\nThe targets provided to the addData function do not match number of outputs mentioned earlier."
            self.data.append({"input": input_array, "target": output})
        elif self.task == "Classification":
            assert isinstance(
                output, str
            ), f"\n\nThe output given to addData for Classification is a {type(output)}.\nExpected a string."
            assert isinstance(
                input_array, list
            ), f"\n\nThe inputs provided to the addData function are of type {type(input_array)}.\nExpected list."
            assert self.inputNodes == len(
                input_array
            ), f"\n\nThe inputs provided to the addData function do not match number of inputs mentioned earlier.\nThe lenght is {len(input_array)}, expected : {self.inputNodes}"
            if not (output in self.labels):
                assert (
                    len(self.labels) < self.outputNodes
                ), f"\n\nCannot add a data point with a new label. The number of labels added are already equal to the number of labels specified while creating the network."
                self.labels.append(output)
            target_array = [0 for _ in range(self.outputNodes)]
            target_array[self.labels.index(output)] = 1
            self.data.append({"input": input_array, "target": target_array})

    # function to add testing data to the network
    def addTestingData(self, input_array, label):
        assert (
            self.task == "Classification"
        ), "\nTesting data can only be added for the task Classification.\n"
        assert isinstance(
            label, str
        ), f"The output given to addTestingData for Classification is a {type(label)}.\nExpected a string."
        assert isinstance(
            input_array, list
        ), f"\n\nThe inputs provided to the addTestingData function are of type {type(input_array)}.\nExpected list."
        assert self.inputNodes == len(
            input_array
        ), f"\n\nThe inputs provided to the addTestingData function do not match number of inputs mentioned earlier.\nThe lenght is {len(input_array)}, expected : {self.inputNodes}"
        assert (
            label in self.labels
        ), f"\n\nThe label is not in the list of labels given to add the data.\nThe list is {self.labels}\nGot label {label}"
        self.testingData.append({"input": input_array, "label": label})

    @staticmethod
    def myRunTest(nn, testing, q):
        correct = 0
        for data in testing:
            inputs = data["input"]
            label = data["label"]
            classification = nn.classify(inputs)
            dataClass = classification["class"]
            if dataClass == label:
                correct += 1
        q.put(correct)

    # method to run tests using testing data
    def runTest(self):
        assert (
            self.task == "Classification"
        ), "\nTests can be run only for the task Classification.\n"
        if len(self.testingData) == 0:
            return
        processes = []
        testing = np.array_split(self.testingData, 5)
        testing = [list(x) for x in testing]
        q = Queue()
        for i in range(5):
            processes.append(Process(target=self.myRunTest, args=[self, testing[i], q]))
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        totalCorrect = 0
        while not q.empty():
            totalCorrect += q.get()
        accuracy = totalCorrect / len(self.testingData) * 100
        error = 100 - accuracy
        wrong = len(self.testingData) - totalCorrect
        del q
        del processes
        del testing
        return {
            "accuracy": accuracy,
            "error": error,
            "correct": totalCorrect,
            "wrong": wrong,
        }

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

    # function to save a compiled model
    def save(self, filename, infoName=""):
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet so it cannot be saved.\n"
        assert not self.isTraining, "\n\nThe model is training and cannot be saved."
        filenameSplit = filename.split(".")
        if filenameSplit[len(filenameSplit) - 1] != "json":
            filenameSplit.append("json")
        mylayers = []
        for i in range(len(self.layers)):
            l = copy.deepcopy(self.layers[i])
            if l.weights is not None:
                l.weights.data = l.weights.data.tolist()
            if l.bias is not None:
                l.bias.data = l.bias.data.tolist()
            mylayers.append(
                {
                    "nodes": l.nodes,
                    "name": l.name,
                    "type": l.type,
                    "key": l.key,
                    "weights": l.weights.__dict__ if l.weights is not None else None,
                    "bias": l.bias.__dict__ if l.bias is not None else None,
                }
            )
        if self.task == "Regression":
            mylayers.insert(
                0,
                {
                    "Contents": "NeuralNetwork Model",
                    "Info_Name": infoName if infoName != "" else "Untitiled",
                    "Config_Info": {
                        "layers": len(self.layers),
                        "layer_nodes": [l.nodes for l in self.layers],
                        "layer_names": [l.name for l in self.layers],
                    },
                },
            )
        else:
            mylayers.insert(
                0,
                {
                    "Contents": "NeuralNetwork Model",
                    "Info_Name": infoName if infoName != "" else "Untitiled",
                    "Config_Info": {
                        "layers": len(self.layers),
                        "layer_nodes": [l.nodes for l in self.layers],
                        "layer_names": [l.name for l in self.layers],
                    },
                    "labels": self.labels,
                },
            )
        try:
            with open(filename, "x", encoding="utf-8") as file:
                mydataJSON = json.dumps(mylayers, ensure_ascii=False, indent=2)
                file.write(mydataJSON)
                file.close()
        except:
            for i in range(1, 10):
                try:
                    myFileName = ""
                    for name in filenameSplit:
                        if name == "json":
                            break
                        myFileName = myFileName + name
                    myFileName = myFileName + f"_({i})" + ".json"
                    with open(myFileName, "x", encoding="utf-8") as file:
                        mydataJSON = json.dumps(mylayers, ensure_ascii=False, indent=2)
                        file.write(mydataJSON)
                        file.close()
                    warnings.warn(
                        f"\n\nA file with the given file name already exists.\nThe name of the file saved right now is now {myFileName}"
                    )
                    return
                except TypeError:
                    assert (
                        False
                    ), "\n\nThere was some error in serialization of something into JSON object...\nCould not save the model."
                except:
                    pass
            assert (
                False
            ), "\n\nFile already exists. Change the file name to save the model.\n"

    # function to load model from a file.
    def load(self, filename):
        assert not self.isTraining, "\n\nThe model is training and cannot be loaded."
        filenameSplit = filename.split(".")
        assert (
            filenameSplit[len(filenameSplit) - 1] == "json"
        ), "\n\nInvalid loading file format.\nCan load only JSON files.\n"
        del filenameSplit
        with open(filename, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except:
                assert False, "\n\nThere was an error loading the file...\n"
            file.close()
        assert isinstance(
            data, list
        ), "\n\nThe data loaded from the file is not of required format."
        assert len(data) - 1 == len(
            self.layers
        ), "\n\nThe number of layers in the loaded data is not equal to the number of layers of this network.\nWrong configuration of data loaded from file.\nError loading the model."
        for i in range(len(data) - 1):
            assert (
                data[i + 1]["key"] == self.layers[i].key
            ), "\n\nThe order of layers in the file is incorrect.\nInvalid configuration of the file."
            assert (
                data[i + 1]["nodes"] == self.layers[i].nodes
            ), f"\n\nThe number of nodes of layer {i} in the data, is not equal to the number of nodes of layer {i} in the network.\nWrong configuration of data loaded from file.\nError loading the model."
        for l in data:
            if not ("Contents" in l.keys()):
                myLayer = self.layer(nodes=l["nodes"], name=l["name"])
                myLayer.type = l["type"]
                myLayer.key = l["key"]

                if l["weights"] is not None and l["bias"] is not None:
                    myLayer.weights = matrix(
                        rows=l["weights"]["rows"],
                        cols=l["weights"]["cols"],
                        name=l["weights"]["name"],
                        wait=True,
                    )
                    myLayer.weights.data = np.array(l["weights"]["data"])
                    myLayer.bias = matrix(
                        rows=l["bias"]["rows"],
                        cols=l["bias"]["cols"],
                        name=l["bias"]["name"],
                        wait=True,
                    )
                    myLayer.bias.data = np.array(l["bias"]["data"])
                self.layers[myLayer.key] = myLayer
            else:
                if self.task == "Classification":
                    self.labels = l["labels"]

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
    def myLossPlotter(queue, endQueue, plottingType):
        anim = None
        xdata = []
        ydata = []
        fig, ax = plt.subplots()
        (In,) = plt.plot([], [])

        def init():
            ax.set_title("Training Process")
            if plottingType == "Data":
                ax.set_xlabel("Data Points")
            else:
                ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss / Cost")
            return (In,)

        def animate(i):
            while not queue.empty():
                queueData = queue.get_nowait()
                xdata.append(queueData[1])
                ydata.append(queueData[0])
            ax.cla()
            init()
            ax.plot(xdata, ydata, linewidth=2, color="green")
            if not endQueue.empty():
                shouldplot = endQueue.get()
                if not shouldplot:
                    anim.event_source.stop()
            return (In,)

        anim = FuncAnimation(fig, animate, init_func=init, blit=True, interval=2)
        plt.style.use("fivethirtyeight")
        fig.set_size_inches(10, 7)
        plt.show()

    @staticmethod
    def trainNotToBeUsed(nn, queue, epochs, plot_interval, lossQueue, debug):
        epochLosses = []
        k = 1
        if debug:
            queue1 = Queue()
            endQueue = Queue()
            if plot_interval < 1:
                plotingProcess = Process(
                    target=NeuralNetwork.myLossPlotter, args=[queue1, endQueue, "Data"]
                )
            else:
                plotingProcess = Process(
                    target=NeuralNetwork.myLossPlotter, args=[queue1, endQueue, "N"]
                )
            plotingProcess.daemon = True
            plotingProcess.start()
        for epochCounter in range(epochs):
            random.shuffle(nn.data)
            # training an epoch
            epochLosses = []
            z = 1
            for d in range(len(nn.data)):
                input_array = nn.data[d]["input"]
                target_array = nn.data[d]["target"]
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
                if plot_interval == 0 and debug:
                    queue1.put([loss, epochCounter * len(nn.data) + d + 1])

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
                if debug and plot_interval > 0 and plot_interval < 1:
                    if d + 1 >= int(plot_interval * len(nn.data)) * z:
                        # should plot
                        queue1.put([loss, epochCounter * len(nn.data) + d + 1])
                        z += 1
            # End of going through all data (End of an EPOCH)
            sum = 0
            for i in range(len(epochLosses)):
                sum += epochLosses[i]
            meanLoss = sum / len(epochLosses)
            if debug and plot_interval >= 1:
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
        if debug:
            endQueue.put(False)
            plotingProcess.join()

    def train(
        self,
        whileTraining,
        epochs,
        plotInterval=5,
        debug=True,
        everyEpoch=False,
    ):
        assert (
            self.compiled
        ), "\n\nThe model is not compiled yet.\n Compile the model to train..\n"
        assert self.isTraining == False, "\n\nThe model is already training."
        assert (
            plotInterval < epochs
        ), "\n\nThe plot interval is less than the number of epochs.\n"

        self.isTraining = True
        queue = Queue()
        lossQueue = Queue()
        trainingProcess = Process(
            target=self.__class__.trainNotToBeUsed,
            args=[self, queue, epochs, plotInterval, lossQueue, debug],
        )
        trainingProcess.daemon = False
        trainingProcess.start()
        while trainingProcess.is_alive():
            if everyEpoch:
                EpochInfo = lossQueue.get()
                if not EpochInfo:
                    break
                whileTraining(EpochInfo[0], EpochInfo[1])
            else:
                whileTraining()
        updatedLayers = queue.get()
        for i in range(1, len(self.layers)):
            self.layers[i] = updatedLayers[i]
        self.isTraining = False

    # Cost / Loss function :
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
        assert (
            self.task == "Regression"
        ), "\n\nTask given is not Regression therefore cannnot predict."
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
        return prediction

    def classify(self, input_array):
        # assert (
        #     len(self.labels) == self.outputNodes
        # ), "\n\nAll the labels have not yet been given.\nTo Classify, provide all the labels to the network."
        assert (
            self.task == "Classification"
        ), f"\n\nTask given is not Classification therefore cannot classify."
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
        myMax = np.NINF
        for num in prediction:
            if num > myMax:
                myMax = num
        labelIndex = prediction.index(myMax)
        if len(self.labels) > labelIndex:
            Class = self.labels[labelIndex]
            return {"class": Class, "confidence": myMax}

    # private function to connect two layers:
    # Connecting = creating matrices of suitable length and initiallzing randomly
    def __connect(self, index_l1, index_l2):
        nodes_l1 = self.layers[index_l1].nodes
        nodes_l2 = self.layers[index_l2].nodes
        self.layers[index_l2].weights = matrix(nodes_l2, nodes_l1, "weights")
        self.layers[index_l2].bias = matrix(nodes_l2, 1, "bias")

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
        self.layers[0].key = 0
        self.layers[len(self.layers) - 1].key = len(self.layers) - 1
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


class matrix:
    def __init__(self, rows, cols, name=None, wait=False):
        self.rows = rows
        self.cols = cols
        self.name = name
        if wait:
            self.data = []
        else:
            self.data = np.random.uniform(-1, 1, (self.rows, self.cols))

    def __str__(self):
        print("\n")
        print(self.data)
        if self.name is None:
            return "Matrix : \n" + f"\tRows: {self.rows}\n" + f"\tCols: {self.cols}\n"
        else:
            return (
                f"Matrix : {self.name}\n"
                + f"\tRows: {self.rows}\n"
                + f"\tCols: {self.cols}\n"
            )

    def mean(self):
        return np.average(self.data)

    def simpleMultiply(self, n):
        if isinstance(n, matrix):
            assert (
                self.rows == n.rows and self.cols == n.cols
            ), "Invalid Matrix Provided"
            self.data = np.multiply(self.data, n.data)
        elif isinstance(n, numbers.Number):
            self.data = self.data * n

    @staticmethod
    def multiply(m1, m2):
        assert (
            m1.cols == m2.rows
        ), f"Cols of m1 are not equal to rows of m2\nCols of m1 are {m1.cols}\nRows of m2 are {m2.rows}"
        if m1.name is None or m2.name is None:
            result = matrix(m1.rows, m2.cols)
        else:
            result = matrix(m1.rows, m2.cols, f"Dot product ({m1.name}.{m2.name})")
        result.data = np.dot(m1.data, m2.data)
        return result

    def toList(self):
        return self.data.flatten().tolist()

    def map(self, fn):
        self.data = fn(self.data)

    @staticmethod
    def map_static(m, fn):
        result = matrix(m.rows, m.cols, f"{m.name} (Mapped)")
        result.data = fn(m.data)
        return result

    @staticmethod
    def toMatrix(a, name=None):
        assert isinstance(a, list), "Invalid Parameters."
        if name is None:
            m = matrix(len(a), 1)
        else:
            m = matrix(len(a), 1, name)
        m.data = np.array(a).reshape(len(a), 1)
        return m

    @staticmethod
    def subtract(a, b):
        assert isinstance(a, matrix) and isinstance(b, matrix), "Invalid Parameters."
        assert a.rows == b.rows and a.cols == b.cols, "Invalid Parameters"
        if a.name is None or b.name is None:
            result = matrix(a.rows, a.cols, "Results")
        else:
            result = matrix(a.rows, a.cols, f"Results({a.name}-{b.name})")
        result.data = a.data - b.data
        return result

    def add(self, n):
        assert isinstance(n, matrix) or isinstance(
            n, numbers.Number
        ), "\n\n\nInvalid Parameters\n"
        if isinstance(n, matrix):
            assert n.rows == self.rows and n.cols == self.cols, "Invalid Parameters"
            self.data = self.data + n.data
        else:
            self.data = self.data + n

    @staticmethod
    def transpose(m):
        if m.name is None:
            result = matrix(m.cols, m.rows, "Result")
        else:
            result = matrix(m.cols, m.rows, f"{m.name} (Transposed)")
        result.data = m.data.T
        return result

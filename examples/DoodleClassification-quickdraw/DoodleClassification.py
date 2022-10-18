# Importing the neural network library
from NeuralNetwork import *
import struct
import time

# Creating dictionaries of clouds, clocks, toothbrushes and trees
# (Things that we want to classify into)
clouds = {"Training": None, "Testing": None, "label": "Cloud"}
clocks = {"Training": None, "Testing": None, "label": "Clock"}
toothbrushes = {"Training": None, "Testing": None, "label": "Toothbrush"}
trees = {"Training": None, "Testing": None, "label": "Tree"}

# Function to prepare the data from the files.
def prepareData(category):
    category["Training"] = []
    category["Testing"] = []
    # Amount of training data.
    # last part of data goes to testing data.
    threshold = 4000
    data = getData(category["label"])

    for i in range(5000):
        offset = i * 784
        if i < threshold:
            category["Training"].append(
                {
                    "imageData": data[offset : offset + 784],
                    "label": category["label"],
                }
            )
        else:
            category["Testing"].append(
                {
                    "imageData": data[offset : offset + 784],
                    "label": category["label"],
                }
            )


# Function to read data from the files.
def getData(Type):
    data = []
    if Type == "Cloud":
        with open("data/cloud10000.bin", "rb") as file:
            byte = file.read(1)
            while byte != b"":
                byte = struct.unpack("B", byte)[0]
                data.append(byte)
                byte = file.read(1)
            file.close()
        return data
    elif Type == "Clock":
        with open("data/clock10000.bin", "rb") as file:
            byte = file.read(1)
            while byte != b"":
                byte = struct.unpack("B", byte)[0]
                data.append(byte)
                byte = file.read(1)
            file.close()
        return data
    elif Type == "Toothbrush":
        with open("data/toothbrush10000.bin", "rb") as file:
            byte = file.read(1)
            while byte != b"":
                byte = struct.unpack("B", byte)[0]
                data.append(byte)
                byte = file.read(1)
            file.close()
        return data
    elif Type == "Tree":
        with open("data/tree10000.bin", "rb") as file:
            byte = file.read(1)
            while byte != b"":
                byte = struct.unpack("B", byte)[0]
                data.append(byte)
                byte = file.read(1)
            file.close()
        return data


# Data normalization function
def normalize(x):
    return x / 255.0


if __name__ == "__main__":
    print("Loading the data...")

    start = time.time()
    # preparing the data
    prepareData(clouds)
    prepareData(clocks)
    prepareData(toothbrushes)
    prepareData(trees)

    # Adding all the data to respective training and testing data lists
    training = (
        clouds["Training"]
        + clocks["Training"]
        + toothbrushes["Training"]
        + trees["Training"]
    )
    testing = (
        clouds["Testing"]
        + clocks["Testing"]
        + toothbrushes["Testing"]
        + trees["Testing"]
    )

    print("Done Preparing data.")
    print("Training and testing data ready.")
    print(f"Length of the testing set is " + str(len(testing)))

    # Creating the NeuralNetwork with 784 inputs, 4 final labels, task being classification
    nn = NeuralNetwork(inputs=784, labels=4, task="Classification")

    # Adding layers to the network.
    nn.addLayer(NeuralNetwork.layer(name="Hidden1", units=128, activation=sigmoid))
    nn.addLayer(NeuralNetwork.layer(name="Hidden2", units=256, activation=tanh))
    nn.addLayer(NeuralNetwork.layer(name="Hidden3", units=72, activation=ReLU))
    nn.addLayer(NeuralNetwork.layer(name="Hidden4", units=16, activation=ReLU))

    # Compiling / finalizing the model
    nn.compileModel(activation=sigmoid)
    print("Compiled the model " + str(nn.compiled))

    # Normalizing and Adding the training data to the network.
    for i in range(len(training)):
        inputs = training[i]["imageData"]
        inputs = map(normalize, inputs)
        label = training[i]["label"]
        inputs = list(inputs)
        nn.addTrainingData(input_array=inputs, output=label)

    print("Added the Data...")
    print("Length of added training data is " + str(len(nn.data)))

    # While Training function to be run when the network is training.
    def whileTraining(epochs, loss):
        print(f"Epoch : {epochs}, Loss : {loss}")

    # Setting the learning rate
    nn.setLearningRate(0.01)
    # starting training of the network.
    nn.train(
        whileTraining=whileTraining,
        epochs=1,
        plotInterval=0.001,
        debug=True,
        # The while training function will be called after every epoch
        everyEpoch=True,
    )

    # Adding the testing data.
    for data in testing:
        inputs = data["imageData"]
        inputs = map(normalize, inputs)
        label = data["label"]
        inputs = list(inputs)
        nn.addTestingData(inputs, label)

    # Running the test.
    print("Starting to run test.")
    results = nn.runTest()
    print(results)

    # Saving out the network parameters.
    nn.save(
        "DoodleClassificationModel.json",
        infoName="Doodle Classification model",
        moreInfo={"Parameters": results},
    )

    # Reloading saved model into another network object.
    nn2 = NeuralNetwork.fromFile("DoodleClassificationModel.json")
    print("This is the new Neural Network object: ", end="")
    print(nn2)

    end = time.time()
    print(f"Completed in {end-start} seconds...")

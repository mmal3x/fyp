from keras.models import load_model
from model import testModel


def main():


    #print("\nRun individual CNN model, Genetic CNN model or GUI: \n")

    # model, scores = mod.trainModel()

    # testing a model on the unseen test data
    model = load_model("INSERT .h5 FILE NAME HERE")
    testModel(model)


if __name__ == "__main__":
    main()
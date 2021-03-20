import os
import time


# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#
import model as mod

# https://stackoverflow.com/questions/43650099/a-switch-case-program-in-python

def main():

    # start = time.time()

    #print("\nRun individual CNN model, Genetic CNN model or GUI: \n")

    mod.trainModel()

    # print("\nRan in {} seconds".format(time.time()-start))


if __name__ == "__main__":
    main()
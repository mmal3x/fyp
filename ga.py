# ga.py

# Author: Alex Waigumo Kabui

# Code Acknowledgments:
# Shaashwat Agrawal - https://shaas2000.medium.com

import random as rand
import matplotlib.pyplot as plt
from model import trainMultiple, final_model, testModel
from keras.models import save_model
import statistics

layers = [0, 2, 6, 7, 8] # the layers in the convnet which contain weights (the filters are the learnable parameters)
model_pop = 5 # defining 10 individuals in the init population
generations = 2 # defining the stopping limit for the number of generations to train and breed
mutation_rate = 0.1 # 10% mutation rate
models = [] # storing each model
crossover_rate = 0.8 # 80% crossover rate

# function to generate the first generation of models
def initialise():

    # getting the first 10 models of the initial population
    for i in range(model_pop):
        models.append(final_model())


# mutation of the weights
# A new model is taken from the new generation
# and random values are assigned to certain weights
# within the boundary.
def mutation(newModel):

    # mutation of randomly selected weights in a newly
    # created model. This is essential for the exploration
    # of the search space. This helps the GA avoid a lack
    # of range (diversity) in the solutions. The higher the mut_rate, the more
    # diversified the population becomes at the expense of convergence.
    # This is where crossover comes in to allow the algorithm to
    # converge more quickly.
    for l in layers:

        # should the random value be less than the mut rate
        # multiply the bias weights by a random number  between
        # -0.5 and 0.5
        for b in range(len(newModel.layers[l].get_weights()[1])):
            m_rand = rand.random()  # getting random value

            if(m_rand < mutation_rate):
                newModel.layers[l].get_weights()[1][b] = newModel.layers[l].get_weights()[1][b] * rand.uniform(-0.5, 0.5)

    # the weights of each of each layer are multiplied by a random value
    # between -0.5 and 0.5.
    for lx in layers:
        for wt in newModel.layers[lx].get_weights()[0]:
            m_rand = rand.random()
            if(m_rand < mutation_rate):
                for ly in range(len(wt)):
                    y_ran = rand.random()
                    if(y_ran < mutation_rate):
                        newModel.layers[lx].get_weights()[0][ly] = newModel.layers[lx].get_weights()[0][ly] * rand.uniform(-0.5, 0.5)


    return newModel

# this function performs the mating process where
# the models with the best weights are paired
def crossover(models):

    newModels = [] # list of new_models for crossover.

    # the first two models in the list are placed in the new models
    newModels.append(models[0])
    newModels.append(models[1])

    for i in range(2, model_pop):

        # the number of parents in the population
        # must be equal or less than the population size
        if i < (model_pop - 2):
            if(i == 2):
                # a random choice of two of the top three models
                # is made when i = 2
                first_parent = rand.choice(models[:3])
                second_parent = rand.choice(models[:3])

            else:
                # when the population no. of parent in the pop
                # is not equal/less than the pop size
                # select any member population member of the
                # population as the parents
                first_parent = rand.choice(models[:])
                second_parent = rand.choice(models[:])

            # should the random number be less than the crossover
            # rate, the bias weights of the parents are swapped to diversify
            # the models by allowing them to access the weights
            # of other models (group consciousness)

            for i in layers:
                c_rand = rand.random()
                if(c_rand < crossover_rate):
                    weightsFirst = first_parent.layers[i].get_weights()[1]

                    # crossing over the bias weights of the second parent to the first parent & vice versa
                    first_parent.layers[i].get_weights()[1] = second_parent.layers[i].get_weights()[1]
                    second_parent.layers[i].get_weights()[1] = weightsFirst

                    newModel = rand.choice([first_parent, second_parent])

        else:
            newModel = rand.choice(models[:])


            # once crossover is complete the new model
            # is ready to be mutated and added to the
            # new models list.
        newModels.append(mutation(newModel))

    return newModels

# here we will select the models with the best
# sorting the fitness values of the models to find
# the best offspring and parents which allows us to
# create new generation of models
def selection(models, fit_score):

    # assigning a fitness value to each model based on its loss figure
    # i.e the model with worst loss is assigned a value of 0 whereas
    # the best model is assigned a value of 9.
    list_asc = sorted(range(len(fit_score)), key=lambda f:fit_score[f])
    f_models = [models[i] for i in list_asc]

    # reversing the model list in ascending order from
    # the model with the worst loss to the best loss
    f_models.reverse()

    # the fitness evaluated models (as per their loss) are now ready for cross
    newModels = crossover(f_models)

    return newModels

# main method calling the initialise method which appends
# the models this loop prints out loss and accuracy info for
# each generation.the

def main():

    global model_pop
    global generations
    global mutation_rate
    global models
    global layers

    # the initialise method appends 10 model architectures into models []
    initialise()

    # saving the avg accuracy scores for each generation
    # and the generation number
    fullClassAccList = []
    fullValidAccList = []
    num_of_gens = []

    for gen_no in range(generations):

        gen_counter = gen_no + 1
        num_of_gens.append(gen_counter)
        # calling the trainMultiple function created in model.py
        # to train many models over x epochs and print the loss information
        models, loss, classAccs, validAccs = trainMultiple(models)
        print(loss)
        lossInfo = ', '.join(str(v) for v in loss)
        print("Generation Number: " + str((gen_counter)) + "\n" + "Training loss Values for each member: \n" + lossInfo)

        # retrieving the avg classification and validation accuracies of all models per generation
        fullClassAccList.append(statistics.mean(classAccs))
        fullValidAccList.append(statistics.mean(validAccs))

        # the trained models and loss info are used to

        # calculate the fitness of the next population
        models = selection(models, loss)


    # getting the best model
    bestModel = []

    # retrieving the model with the highest validation accuracy
    sortedValAccs = sorted(validAccs)
    finalValidAccs = sorted(range(len(validAccs)), key=lambda i:validAccs[i])
    finalModels = [models[i] for i in finalValidAccs]

    finalModels.reverse() # best model comes first

    # converting validation accuracies to string representation
    validInfo = ', '.join(str(v) for v in sortedValAccs)
    print("Validation Accuracies of the final 10 models: \n" + validInfo)
    print("Average validation accuracy of all final pop models: " + str(statistics.mean(sortedValAccs)))

    # appending the best model to a variable
    bestModel = finalModels[0]

    # subplot displaying and validation loss in terms of the training and validation sets
    # plt.subplot(1, 2, 1)
    # plt.title('Cross Entropy Loss')
    # plt.plot(finalScores.history['loss'], color='red', label='train')
    # plt.plot(finalScores.history['val_loss'], color='purple', label='test')

    # subplot displaying classification accuracy in terms of the training and validation sets
    plt.subplot(1, 2, 2)
    plt.title('Avg Classification/Validation Accuracy')
    plt.plot(fullClassAccList, color='red', label='train')
    plt.plot(fullValidAccList, color='purple', label='test')
    plt.ylabel('accuracy (%)')
    plt.xlabel('generation')
    #plt.xticks(gen_counter)
    plt.show()

    # testing the best model on unseen data
    testModel(bestModel)

    # saving best model to a h5 file
    bestModel.save('best_model.h5')


if __name__ == "__main__":
    main()
import model as mod
import random as rand

#
# https://www.nist.gov/itl/products-and-services/emnist-dataset

model_pop = 10
generations = 15
mutation_rate = 0.1 # 10% mutation rate
models = [] # storing each model
crossover_rate = 0.8 # 80% crossover rate  ***HERE: INCOPORATE THIS INTO CROSSOVER


# function to generate the first generation of models
def initialise():

    for i in range(model_pop):
        models.append(mod.final_model())


# mutation of the weights
# A new model is taken from the new generation
# and random values are assigned to certain weights
# within the boundary.
def mutation(newModel):
    weights = []
    for weight in newModel:
        x = rand.random()

        if(x < mutation_rate):
            weights.append(rand.random())

        else:
            weights.append(weight)

        weights.append(rand.random())

    return weights

# this function performs the mating process where
# the models with the best weights are paired
def crossover(models):

    newModels = []
    # the two new models (offspring) are saved in the first two positions of the list
    newModels.append(models[0])
    newModels.append(models[1])

    for i in range(2, model_pop):

        newModel = []

        # the number of parents in the population
        # must be even or less than the population size
        if(i < (model_pop - 2)):
            if(i == 2):
                # if the population is ***
                first_parent = rand.choice(models[:3])
                second_parent = rand.choice(models[:3])

            else:
                first_parent = rand.choice(models[:])
                second_parent = rand.choice(models[:])

            for i in range(len(first_parent)):
                n = rand.random()
                if(n < 0.5):
                    newModel.append(first_parent[i])

                else:
                    newModel.append(second_parent[i])

        else:
            newModel = rand.choice(models[:])


        # once crossover is complete the new model
        # is ready to be mutated and added to the
        # new model list.
        newModels.append(mutation(newModel))

    return newModels

# here we will select the models with the best
# sorting the fitness values of the models to find
# the best offspring and parents which allows us to
# create new generation of models
def selection(models, fit_score):
    sorted_list = sorted(range(len(fit_score)), key=lambda x:fit_score[x])

    models = [models[i] for i in sorted_list]

    models.reverse()

    # if the termination condition is not satisfied
    # after the fitness calculation we breed the
    # models with the best fitness.
    newModels = crossover(models)
    return models

# firstly, the initialise method is called which appends
# # the models this loop prints out loss and accuracy info for
# each generation.the

initialise()

for gen in range(generations):

    # calling the modelTrain function created in model.py
    trainedModels, lossInfo = mod.modelTrain(models)
    print(lossInfo)

    # the trained models and loss info are used to
    # selection the next population
    models = selection(trainedModels, lossInfo)










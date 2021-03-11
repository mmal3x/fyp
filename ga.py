# import model as mod
import random as rand
from model import trainMultiple, final_model
from keras.models import save_model

layers = [0, 2, 6, 7, 8] # the layers in the convnet which contain weights (the filters are the learnable parameters)
model_pop = 10 # defining 10 individuals in the init population
generations = 2 # defining the stopping limit for the number of generations to train and breed
mutation_rate = 0.1 # 10% mutation rate
models = [] # storing each model
crossover_rate = 0.8 # 80% crossover rate  ***HERE: INCOPORATE THIS INTO CROSSOVER

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
    # weights = []

    # mutation of randomly selected weights in a newly
    # created model. This is essential for the convergence
    # of the GA.
    for l in layers:

        # should the random value be less than the mut rate
        # multiply the bias weights by a random number  between
        # -0.5 and 0.5
        for b in range(len(newModel.layers[l].get_weights()[1])):
            x_ran = rand.random()  # getting random value

            if(x_ran < mutation_rate):
                newModel.layers[l].get_weights()[1][b] = newModel.layers[l].get_weights()[1][b] * rand.uniform(-0.5, 0.5)

    # the weights of each of each layer are
    for lx in layers:
        for wt in newModel.layers[lx].get_weights()[0]:
            x_ran = rand.random()
            if(x_ran < mutation_rate):
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

            for i in layers:
                weightsFirst = first_parent.layers[i].get_weights()[1]

                # crossing over the weights of the second parent to the first parent & vice versa
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

    for gen_no in range(generations):

        # calling the trainMultiple function created in model.py
        # to train many models over x epochs and print the loss information
        models, lossInfo = trainMultiple(models)
        print(lossInfo)
        stringInfo = ', '.join(str(v) for v in lossInfo)
        print("Generation Number: " + str((gen_no + 1)) + "\n" + "Loss Values for each member: " + stringInfo)

        # the trained models and loss info are used to
        # select the next population
        models = selection(models, lossInfo)


    # getting the best model
    bestModel = []

    # retrieving the model with the lowest loss by assigning a fitness
    # value to all the models in order to order them accordingly in
    # the model list
    finalLosses = sorted(range(len(lossInfo)), key=lambda i:lossInfo[i])
    finalModels = [models[i] for i in finalLosses]

    print(finalLosses[1])

    bestModel.append(finalModels[0])
    bestM = bestModel.pop()

    bestM.save('best_model.h5')



if __name__ == "__main__":
    main()










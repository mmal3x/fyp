from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder


#### Data Preparation - Load data into train and test sets ####
# loading the training and test data into pandas dataframes
train_dataset = pd.read_csv('mnist_train.csv')
test_dataset = pd.read_csv('mnist_test.csv')

# separating the y label (the class) from the train & test datasets
# in order to validate and evaluate the model
train_ylabels = train_dataset['label']
test_ylabels = test_dataset['label']

# dropping the label column from the training & test dataset
train = train_dataset.drop(labels = ["label"], axis = 1)
test = test_dataset.drop(labels = ["label"], axis = 1)

#------------------------------------------------------------------------
# countplot of y training labels. As this data is labelled
# this is a form of supervised learning. The labels are needed
# for the model to learn the training set. The graph shows that
# the number of digits are evenly distributed throughout the
# dataset.

# g = sb.countplot(train_ylabels)
#-----------------------------------------------------------------------
# ensuring there are no missing values in the training data
train.isnull().any().describe()

# ensuring there are no missing values in the test data
test.isnull().any().describe()
#----------------------------------------------------------------------
#### Normalising the training and test data between 0-1. ######
# the image pixels are all in the range 0-255. To normalise them
# both train and test sets are divided by 255.
train = train / 255
test = test / 255

#---------------------------------------------------------------------
#### Reshaping the image data using height, width & depth #######
# reshaping the training set to the dimensions...
# [samples][width][height][channels]
train = train.values.reshape((train.shape[0], 28, 28, 1)).astype('float32')
test = test.values.reshape((test.shape[0], 28, 28, 1)).astype('float32')

#------------------------------------------------------------------------
##### Encoding the labels for the output layer ####
# use one hot encoding in order to classify the outputs
# the outputs are set to [1,0,0,0,0,0,0,0,0,0] binary vectors
train_ylabels = np_utils.to_categorical(train_ylabels, num_classes = 10)
# print(train_ylabels.shape)

#-----------------------------------------------------------------------------

### Setting the random seed to 2. The random seed helps with the model's reproducibility of results.
random_seed = 2
test_perc = 0.15

# Splitting the training & validation sets to evaluate the
# performance of the model. The validation size can be adjusted in accordance
# with the test size. The val_x and val_y params are used to validate the
# train and hold out sets.
train, val_x, train_ylabels, val_y = train_test_split(train,
                                                         train_ylabels,
                                                         test_size = test_perc,
                                                         random_state = random_seed)


# example output of a digit in the dataset.
g = plt.imshow(train[10][:,:,0], cmap = 'Greys')

def final_model():

    modelInstance = Sequential()

    # two convolutional layers with sizes 30. The ReLu activation function
    # is used for both of these layers. A max pooling layer
    # removes noisy activations the first two CNN layers.
    modelInstance.add(Conv2D(filters = 30, kernel_size = (5,5),
                     input_shape = (28, 28, 1),
                     activation = 'relu'))

    # the maxpooling layer downsamples the noisy the activations
    # using a 2x2 filter
    modelInstance.add(MaxPooling2D(pool_size = (2,2)))

    # second layer of convolution containing 15 layers and 3x3 filters
    modelInstance.add(Conv2D(filters = 15, kernel_size = (3,3),
                     activation = 'relu'))

    modelInstance.add(MaxPooling2D(pool_size = (2,2)))

    # drop-out is a regularisation technique to reduce the model
    # overfitting on the training data. A random % of the
    # activations are kept while the others are dropped as their
    # weights are zeroed out.
    modelInstance.add(Dropout(0.3))

    # the flatten layer converts the feature maps of the convolutional
    # layer into vector representatation for the dense layer to
    # further learn the local features.
    modelInstance.add(Flatten())

    # three fully-connected layers are used once the layers have been
    # flattened to provide the the  classifier w/ learnable
    # inputs from the pixels. The activation functions relu and
    # softmax are used to add nonlinearity to the model
    # and subsequently optimise the model.
    modelInstance.add(Dense(256, activation = 'relu'))
    modelInstance.add(Dense(100, activation = 'relu'))
    modelInstance.add(Dense(10, activation = 'softmax'))


    # Compiling the model. Cross categorical entropy has been used
    # because a digit can be written in an ambiguous way which
    # introduces uncertainty to the classification. This means
    # soft probabilities by way of binary vectors are required as the
    # output.

    modelInstance.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', metrics = ['accuracy'])


    return modelInstance

#-------------------------------------------------------------------------------
# simple early stopping to stop the training once the model
# starts to overfit on the training data.
es = EarlyStopping(monitor='val_loss', patience = 10, mode='min', verbose=1)

#----------------------------------------------------------------------------
#### Image Data Augmentation ####

# the zoom range is set to 10% to randomly
# zoom some images in the training set
# 10 % has been selected as the mnist digits already have
# a normalised orientation

# the rotation range is set to a particular percentage
# of an image's width. This is done at random.

datagen = ImageDataGenerator(zoom_range = 0.1, rotation_range = 10)

# applying image data augmentation to the training set.
datagen.fit(train)

#-------------------------------------------------------------------------------
#### Function to train a single model ####

def trainModel():
    # saving a single model instance
    model = final_model()

    # Executing the the training for a single model. The scores and the loss info are displayed
    # live and at the end of the training process for each epoch.
    # training the model for 30 epochs with a batch size of 200, callbacks with a
    # patience of 10 and the image data generator which extends the size of the train set
    scores = model.fit_generator(datagen.flow(train,
                                              train_ylabels,
                                              batch_size = 100),
                                 validation_data = (val_x, val_y),
                                 epochs = 1, verbose = 2)


    # getting the mean validation accuracy across all epochs
    print("Average validation accuracy across all epochs: " + str(np.mean(scores.history['val_accuracy'])))


    # subplot displaying and validation loss in terms of the training and validation sets
    plt.subplot(1, 2, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(scores.history['loss'], color='red', label='train')
    plt.plot(scores.history['val_loss'], color='purple', label='test')

    # subplot displaying classification accuracy in terms of the training and validation sets
    plt.subplot(1, 2, 2)
    plt.title('Training vs Validation Accuracy')
    plt.plot(scores.history['accuracy'], color='red', label='train')
    plt.plot(scores.history['val_accuracy'], color='purple', label='test')
    plt.show()

    return model

#--------------------------------------------------------------------------------------
# function where we will save each model after
# they are run them for a maximum of 3 epochs. The function
# returns the model and its respective loss.
def trainMultiple(allModels):

    # list of losses is stored here
    allLosses = []
    allValidationAccs = []

    # the fit_generator function trains
    # the on training data for 3 epochs according
    # to the train_ylabels variable
    for i in range(len(allModels)):

        scores = allModels[i].fit_generator(datagen.flow(train, train_ylabels, batch_size = 200),
                                            validation_data = (val_x, val_y),
                                            epochs = 1, verbose = 2)

        allLosses.append(round(scores.history['loss'][-1], 4))
        allValidationAccs.append(round(scores.history['val_accuracy'][-1],4))

    return allModels, allLosses, allValidationAccs

#-------------------------------------------------------------------------------------

# the evaluate function takes three parameters -
# the test_y data and the predictions of the model on the test data,
# as well as the labels to build the confusion matrix.
def evaluate(actual, predicted, labels):

    # precision, recall, f1 and support scores classification scores.

    pre, rec, f1_score, inst = precision_recall_fscore_support(actual, predicted)
    tot_pre = np.average(pre, weights=inst)
    tot_rec = np.average(rec, weights=inst)
    tot_f1 = np.average(f1_score, weights=inst)
    tot_sup = np.sum(inst)
    # output table 1
    dict1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': pre,
        u'Recall': rec,
        u'F1': f1_score,
        u'Support': inst
    })
    # output table 2
    dict2 = pd.DataFrame({
        u'Label': [u'Average'],
        u'Precision': [tot_pre],
        u'Recall': [tot_rec],
        u'F1': [tot_f1],
        u'Support': [tot_sup]
    })

    dict2.index = [99]

    # concatenating the two dictionaries
    dictionary = pd.concat([dict1, dict2])

    # output table 2
    confusion_mat = pd.DataFrame(confusion_matrix(actual, predicted), columns=labels, index=labels)

    return confusion_mat, dictionary[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]


def main():

    # returning the trained model and the binary label vectors
    model = trainModel() # saving trained model

    # labels for each digit in the test dataset converted from a df column
    # to a list
    y_true = test_dataset['label'].tolist()
    print(y_true)

    # label encoding the test data labels so the confusion matrix can
    # be calculated
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y_true)
    test_y = y[len(test)-1]

    # using the label encoder to retrieve the labels for each class.
    classLabels = labelEncoder.classes_

    # using the trained model to make predictions on the unseen test set
    y_preds = model.predict(test) # this is printing the softmax probabilities

    # converting the predictions to binary vectors in order to compare the predicted
    # values w/ the actual values. val_y is the expected output of the val_x input
    predicted_classes = np.argmax(y_preds, axis=1)
    print(predicted_classes.tolist())

    # evaluating the model on the test data find out the strength of the model
    # on unseen data.
    conf_matrix, evalScores = evaluate(y_true, predicted_classes, classLabels)
    print(conf_matrix)
    print(evalScores)

    # plotting heatmap of confusion matrix.
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt= ".1f")
    plt.show()


if __name__ == "__main__":
    main()
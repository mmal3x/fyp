from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


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

# print(train_ylabels)

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
    model = final_model()

    # training the model for 30 epochs with a batch size of 200, callbacks with a
    # patience of 10 and the image data generator which extends the size of the train set
    scores = model.fit_generator(datagen.flow(train,
                                              train_ylabels,
                                              batch_size = 100),
                                 validation_data = (val_x, val_y),
                                 epochs = 3, verbose = 2)


    # getting the mean validation accuracy across all epochs
    print(np.mean(scores.history['val_accuracy'])) # getting the avg validation loss

    # Executing the the training for a single model. The scores and the loss info are displayed
    # live and at the end of the training process for each epoch.

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

# using trained model to predict on the test set

def evaluate(y_true, y_predictions, labels):
    # precision, recall, f1 and support scores classification scores.

    # y_true = 423
    # y_predictions = 423

    pr, rec, f1, sup = precision_recall_fscore_support(y_true, y_predictions)
    total_pr = np.average(pr, weights=sup)
    total_rec = np.average(rec, weights=sup)
    total_f1 = np.average(f1, weights=sup)
    total_sup = np.sum(sup)
    # output table 1
    dict1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': pr,
        u'Recall': rec,
        u'F1': f1,
        u'Support': sup
    })
    # output table 2
    dict2 = pd.DataFrame({
        u'Label': [u'overall'],
        u'Precision': [total_pr],
        u'Recall': [total_rec],
        u'F1': [total_f1],
        u'Support': [total_sup]
    })

    dict2.index = [999]

    # concatenating the two dictionaries
    dictionary = pd.concat([dict1, dict2])

    # output table 2
    confusion_mat = pd.DataFrame(confusion_matrix(y_true, y_predictions), columns=labels, index=labels)

    return confusion_mat, dictionary[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]


def main():
    # returning the trained model and the binary label vectors
    model = trainModel() # saving trained model
    print(len(test_ylabels)) # actual outputs

    # using the model to make predictions on the unseen test set
    y_preds = model.predict(test)

    # convert the predictions to binary vectors in order to compare the predicted
    # values w/ the actual values. val_y is the expected output of the val_x input
    predicted_classes = np.argmax(y_preds, axis=1)

if __name__ == "__main__":
    main()
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import load_model
from PIL import ImageGrab, Image

#### Data Preparation - Load data into train and test sets ####
# loading the training and test data into pandas dataframes
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

# separating the y label from the original dataset
train_ylabels = train['label']
# dropping the label column from the training & test datasets
train_xx = train.drop(labels = ["label"], axis = 1)
test_xx = test.drop(labels = ["label"], axis = 1)
# test_xx

#------------------------------------------------------------------------
# countplot of y training labels. As this data is labelled
# this is a form of supervised learning. The labels are needed
# for the model to learn the training set. The graph shows that
# the number of digits are evenly distributed throughout the
# dataset.

# g = sb.countplot(train_ylabels)
#-----------------------------------------------------------------------
# ensuring there are no missing values in the training data
train_xx.isnull().any().describe()

# ensuring there are no missing values in the test data
test_xx.isnull().any().describe()
#----------------------------------------------------------------------
#### Normalising the training and test data between 0-1. ######
# the image pixels are all in the range 0-255. To normalise them
# both train and test sets are divided by 255.
train_xx = train_xx / 255
test_xx = test_xx / 255

#---------------------------------------------------------------------
#### Reshaping the image data using height, width & depth #######
# reshaping the training set to the dimensions...
# [samples][width][height][channels]
train_xx = train_xx.values.reshape((train_xx.shape[0], 28, 28, 1)).astype('float32')
test_xx = test_xx.values.reshape((test.shape[0], 28, 28, 1)).astype('float32')

#------------------------------------------------------------------------
##### Encoding the labels for the output layer ####
# use one hot encoding in order to classify the outputs
# the outputs are set to [1,0,0,0,0,0,0,0,0,0] binary vectors
train_ylabels = np_utils.to_categorical(train_ylabels, num_classes = 10)
train_ylabels

#-----------------------------------------------------------------------------

### Setting the random seed to 2. The random seed helps with the model's reproducibility of results.
random_seed = 2

from sklearn.model_selection import train_test_split
# Splitting the training & validation sets to evaluate the
# performance of the model by fitting the model. The test set includes 10,500 images - the training set includes 59,500 images
train_xx, val_x, train_ylabels, val_y = train_test_split(train_xx,
                                                         train_ylabels,
                                                         test_size = 0.15,
                                                         random_state = random_seed)


# example output of a digit in the dataset.
g = plt.imshow(train_xx[10][:,:,0], cmap = 'Greys')

def final_model():

    model = Sequential()

    # two convolutional layers with sizes 30. The ReLu activation function
    # is used for both of these layers. A max pooling layer
    # removes noisy activations the first two CNN layers.
    model.add(Conv2D(filters = 30, kernel_size = (5,5),
                     input_shape = (28, 28, 1),
                     activation = 'relu'))

    # the maxpooling layer downsamples the noisy the activations
    # using a 2x2 filter
    model.add(MaxPooling2D(pool_size = (2,2)))


    model.add(Conv2D(filters = 15, kernel_size = (3,3),
                     activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    # drop-out is a regularisation technique to reduce the model
    # overfitting on the training data. A random % of the
    # activations are kept while the others are dropped as their
    # weights are zeroed out.
    model.add(Dropout(0.3))

    # the flatten layer converts the feature maps of the convolutional
    # layer into vector representatation for the dense layer to
    # further learn the local features.
    model.add(Flatten())

    # three fully-connected layers are used once the layers have been
    # flattened to provide the the  classifier w/ learnable
    # inputs from the pixels. The activation functions relu and
    # softmax are used to add nonlinearity to the model
    # and subsequently optimise the model.
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))


    # Compiling the model. Cross categorical entropy has been used
    # because the a digit can be written in a way where it is hard
    # to tell what it is. This uncertainty means soft probabilities
    # are required as the output.


    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', metrics = ['accuracy'])


    return model

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

datagen = ImageDataGenerator(zoom_range = 0.1,
                             rotation_range = 10)


datagen.fit(train_xx)

#-------------------------------------------------------------------------------
#### Fit the model ####
model = final_model()
# model = load_model('final_iter1.h5')

scores = model.fit_generator(datagen.flow(train_xx, train_ylabels, batch_size = 200),

                             validation_data = (val_x, val_y),
                             epochs = 30, verbose = 2, callbacks = [es])


#-------------------------------------------------------------------------------------
# subplot displaying and validation loss in terms of the training and validation sets
plt.subplot(1, 2, 1)
plt.title('Cross Entropy Loss')
plt.plot(scores.history['loss'], color='red', label='train')
plt.plot(scores.history['val_loss'], color='purple', label='test')

# subplot displaying classification accuracy in terms of the training and validation sets
plt.subplot(1, 2, 2)
plt.title('Classification Accuracy')
plt.plot(scores.history['accuracy'], color='red', label='train')
plt.plot(scores.history['val_accuracy'], color='purple', label='test')
plt.show()


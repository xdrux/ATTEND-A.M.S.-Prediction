import os
import pickle
import time

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Specifying the folder where images are present
TrainingImagePath = 'C:/Users/ANDREAU/Documents/school/up/FOURTH/second/sp2/program/preditionbackend/models/CMSC 57 X3L 2nd Semester 2023-2024'

folder_name = os.path.basename(TrainingImagePath)

# Specify the folder where you want to save your files
models_folder = 'C:/Users/ANDREAU/Documents/school/up/FOURTH/second/sp2/program/preditionbackend/models/'

# Save the model with the folder name as the filename in the models folder
model_filename = os.path.join(models_folder, folder_name + ".h5")

# Save the pickle file in the models folder
map_filename = os.path.join(models_folder, folder_name + "_map.pkl")

# Defining pre-processing transformations on raw images of training data
train_datagen = ImageDataGenerator(
    shear_range=0.1,
    zoom_range=0.1,
    
    horizontal_flip=True,
)

# Defining pre-processing transformations on raw images of testing data
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical')

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical')

# Printing class labels for each face
test_set.class_indices

# Creating lookup table for all faces
TrainClasses = training_set.class_indices

# Storing the face and the numeric tag for future reference
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName

# Saving the face map for future reference
with open(map_filename, 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons = len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

# Initializing the Convolutional Neural Network
classifier = Sequential()

# Adding the first layer of CNN
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(300, 300, 3), activation='relu'))

# Adding the first Max Pooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Additional layer of Convolution for better accuracy
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

# Adding the second Max Pooling layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Flattening the layer
classifier.add(Flatten())

# Fully Connected Neural Network
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))

# Compiling the CNN
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Measuring the time taken by the model to train
StartTime = time.time()

# Training the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=10,
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Saving the trained model
classifier.save(model_filename)
print("Model saved as:", model_filename)

EndTime = time.time()
print("Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes')

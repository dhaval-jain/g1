TRAIN_PATH = "C:/Users/meow/PycharmProjects/project1/CovidDataset/Train"  # gets the the paths in that folder
VAL_PATH = "C:/Users/meow/PycharmProjects/project1/CovidDataset/Val"

# import the libraries as shown below

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE = [224, 224]


resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('C:/Users/meow/PycharmProjects/project1/CovidDataset/Train/*')

# our layers - you can add more if you want
x = Flatten()(resnet.output)


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# view the structure of the model
model.summary()


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('C:/Users/meow/PycharmProjects/project1/CovidDataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 2,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('C:/Users/meow/PycharmProjects/project1/CovidDataset/Val',
                                            target_size = (224, 224),
                                            batch_size = 2,
                                            class_mode = 'categorical')


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train_accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_accuracy')


# save it as a h5 file

import tensorflow as tf

from keras.models import load_model

model.save('model_resnet.h5')

model_json = model.to_json()
with open('model_adam_resnet_2020.json', 'w') as json_file:
    json_file.write(model_json)

print('Model saved to the disk.')
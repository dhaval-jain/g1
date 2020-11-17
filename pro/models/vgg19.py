TRAIN_PATH = "C:/Users/meow/PycharmProjects/project1/CovidDataset1/Train"  # gets the the paths in that folder
VAL_PATH = "C:/Users/meow/PycharmProjects/project1/CovidDataset1/Val"

# import the libraries as shown below

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



IMAGE_SIZE = [224, 224]


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('C:/Users/meow/PycharmProjects/project1/CovidDataset1/Train/*')


# our layers - you can add more if you want
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)



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
training_set = train_datagen.flow_from_directory('C:/Users/meow/PycharmProjects/project1/CovidDataset1/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 5,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('C:/Users/meow/PycharmProjects/project1/CovidDataset1/Val',
                                            target_size = (224, 224),
                                            batch_size = 5,
                                            class_mode = 'categorical')




# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=3,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)




# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss_vgg19_599')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_accuracy_vgg19_599')


# save it as a h5 file

import tensorflow as tf

from keras.models import load_model

model.save('model_vgg19_599.h5')

model_json = model.to_json()
with open('model_adam_vgg19_2020_599.json', 'w') as json_file:
    json_file.write(model_json)

print('Model saved to the disk.')
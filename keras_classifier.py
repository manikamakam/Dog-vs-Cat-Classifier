# encoding=utf8

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1000)




# sample_training_images, _ = next(train_data_gen)

# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# plotImages(sample_training_images[:5])


# model = Sequential([
# 	# 1st Convolutional Layer
#     Conv2D(filters=96, input_shape=inputShape, kernel_size=(11,11), strides=(4,4), padding='same', kernel_regularizer = l2(0.0002)),
#     Activation('relu'),
# 	# Max Pooling
#     MaxPooling2D(pool_size=(3,3), strides=(2,2)),
#     Dropout(0.25),

# 	# 2nd Convolutional Layer
# 	Conv2D(filters=256, kernel_size=(5,5), padding='same',kernel_regularizer = l2(0.0002)),
# 	Activation('relu'),
# 	# Max Pooling
# 	MaxPooling2D(pool_size=(3,3), strides=(2,2)),
# 	Dropout(0.25),

# 	# 3rd Convolutional Layer
# 	Conv2D(filters=384, kernel_size=(3,3), padding='same', kernel_regularizer = l2(0.0002)),
# 	Activation('relu'),

# 	# 4th Convolutional Layer
# 	Conv2D(filters=384, kernel_size=(3,3), padding='same', kernel_regularizer = l2(0.0002)),
# 	Activation('relu'),

# 	# 5th Convolutional Layer
# 	Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_regularizer = l2(0.0002)),
# 	Activation('relu'),
# 	# Max Pooling
# 	MaxPooling2D(pool_size=(3,3), strides=(2,2)),
# 	Dropout(0.25),

# 	# Passing it to a Fully Connected layer
# 	Flatten(),
# 	# 1st Fully Connected Layer
# 	Dense(4096, kernel_regularizer = l2(0.0002)),
# 	Activation('relu'),
# 	# Add Dropout to prevent overfitting
# 	Dropout(0.5),

# 	# # 2nd Fully Connected Layer
# 	# model.add(Dense(4096))
# 	# model.add(Activation(‘relu’))
# 	# # Add Dropout
# 	# model.add(Dropout(0.4))

# 	# # 3rd Fully Connected Layer
# 	# model.add(Dense(1000))
# 	# model.add(Activation(‘relu’))
# 	# # Add Dropout
# 	# model.add(Dropout(0.4))

# 	# Output Layer
# 	Dense(1, kernel_regularizer = l2(0.0002)),
# 	# Activation('softmax')
# ])

def define_model(inputShape):

	model = Sequential([
		Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform',
	           input_shape=inputShape),
	    # BatchNormalization(),
	    MaxPooling2D((2,2)),
	    # Dropout(0.2),

	    Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform'),
	    # BatchNormalization(),
	    MaxPooling2D((2,2)),
	    # Dropout(0.2),

	    Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform'),
	    # BatchNormalization(),
	    MaxPooling2D((2,2)),
	    # Dropout(0.2),

	    Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform'),
	    # BatchNormalization(),
	    MaxPooling2D((2,2)),
	    # Dropout(0.2),

	    Flatten(),
	    Dense(128, activation='relu', kernel_initializer='he_uniform'),
	    # BatchNormalization(),
	    # Dropout(0.5),
	    Dense(1, activation = 'sigmoid')
	])
	opt = Adam(lr=0.001)

	model.compile(optimizer=opt,
	              loss='binary_crossentropy',
	              metrics=['accuracy'])
	return model
# model.summary()

def summarize_diagnostics(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss=history.history['loss']
	val_loss=history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	# plt.show()
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()



train_dir = "./data/train/"
validation_dir = "./data/val"

train_cats_dir = os.path.join(train_dir, 'cats')  
train_dogs_dir = os.path.join(train_dir, 'dogs')  
validation_cats_dir = os.path.join(validation_dir, 'cats')  
validation_dogs_dir = os.path.join(validation_dir, 'dogs') 

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 64
epochs = 40
IMG_HEIGHT = 200
IMG_WIDTH = 200
depth = 3
inputShape = (IMG_HEIGHT, IMG_WIDTH, depth)

model = define_model(inputShape)

train_image_generator = ImageDataGenerator(
					# rescale=1./255,
     #                rotation_range=45,
     #                width_shift_range=.15,
     #                height_shift_range=.15,
     #                horizontal_flip=True,
     #                zoom_range=0.5
     				    rescale = 1./255,
					    rotation_range = 20,
					    width_shift_range = 0.2,
					    height_shift_range = 0.2,
					    shear_range = 0.2,
					    zoom_range = 0.2,
					    fill_mode = 'nearest',
					    horizontal_flip = True)                 # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

print(len(train_data_gen))
print(len(val_data_gen))

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=len(val_data_gen)
)

_, acc = model.evaluate_generator(val_data_gen, steps=len(val_data_gen), verbose=0)
print('> %.3f' % (acc * 100.0))
model.save('model2.h5')

summarize_diagnostics(history)
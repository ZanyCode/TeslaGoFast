from PIL import ImageFont, ImageDraw, Image
from os.path import abspath, join, dirname
import os

from tensorflow.python.keras.preprocessing.image import DirectoryIterator
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
import random
import numpy as np
import cv2
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.lib.utils import info
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
input_shape = (128, 128, 3)
values = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130)
num_classes = len(values)
data_dir = join(DIR_BACKEND, 'train_data_traffic_signs')

def get_traffic_sign_dataset() -> DirectoryIterator:
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        brightness_range=(0.4, 1),
        zoom_range=(0.8, 1.2),
        validation_split=0.2)

    train_dataset = datagen.flow_from_directory(data_dir, target_size=(128, 128), color_mode = 'rgb', class_mode='categorical', batch_size = 128)
    return train_dataset

def get_traffic_sign_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=None,
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=num_classes,
        classifier_activation="softmax"
    )

    locked_layers = 200
    for layer in base_model.layers[:locked_layers]:
        layer.trainable=False
    for layer in base_model.layers[locked_layers:]:
        layer.trainable=True
    # base_model.trainable = False

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training = False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    # x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
    # x=tf.keras.layers.Dropout(0.2)(x)
    x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
    preds=tf.keras.layers.Dense(num_classes,activation='softmax')(x) #final layer with softmax activation
    model = tf.keras.Model(inputs, preds)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss="categorical_crossentropy",
#               metrics=['accuracy'])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == '__main__':
    base_learning_rate = 0.001
    checkpoint_filepath = join(DIR_BACKEND, 'checkpoints_traffic_signs')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        verbose =2,
        mode='min',
        save_best_only=True)

    train_dataset = get_traffic_sign_dataset()
    model = get_traffic_sign_model()


    model.fit(x = train_dataset, epochs=1000, callbacks=[model_checkpoint_callback])


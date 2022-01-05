from PIL import ImageFont, ImageDraw, Image
from os.path import abspath, join, dirname
import os

from tensorflow._api.v2 import data
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
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
input_shape = (67, 119, 3)
value_range = (5, 150)
num_classes = value_range[1] - value_range[0] + 1

def get_number_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=None,
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        # weights=None,
        input_tensor=None,
        pooling=None,
        classes=num_classes,
        classifier_activation="softmax"
    )

    base_model.trainable = False

    # locked_layers = 17
    # for layer in base_model.layers[:locked_layers]:
    #     layer.trainable=False
    # for layer in base_model.layers[locked_layers:]:
    #     layer.trainable=True

    # model = tf.keras.Sequential([
    #     base_model,
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(1024,activation='relu'),
    #     tf.keras.layers.Dense(512,activation='relu'),
    #     tf.keras.layers.Dense(num_classes,activation='softmax')
    # ])

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training = False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    # x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
    x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
    preds=tf.keras.layers.Dense(num_classes,activation='softmax')(x) #final layer with softmax activation
    model = tf.keras.Model(inputs, preds)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def get_number_dataset(data_dir):
    # train_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, labels='inferred', batch_size = 128, image_size=(70, 35), color_mode = 'rgb', label_mode = 'categorical')
    # AUTOTUNE = tf.data.AUTOTUNE

    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # return train_dataset

    def prep_fn(img):
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2
        return img

    # datagen = ImageDataGenerator(
    #     preprocessing_function=prep_fn,
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=0,
    #     width_shift_range=0,
    #     height_shift_range=0,
    #     horizontal_flip=0,
    #     brightness_range=None,
    #     zoom_range=0,
    #     validation_split=0.2)

    datagen = ImageDataGenerator(
        preprocessing_function=prep_fn,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        brightness_range=(0.8, 1),
        zoom_range=(0.8, 1.2),
        validation_split=0.2)

    train_dataset = datagen.flow_from_directory(data_dir, target_size=(67, 119), color_mode = 'rgb', class_mode='categorical', batch_size = 128)
    return train_dataset


if __name__ == '__main__':
    data_dir = join(DIR_BACKEND, 'train_data_numbers')
   

    model = get_number_model()
    train_dataset = get_number_dataset(data_dir)

    # base_learning_rate = 0.001
    checkpoint_filepath = join(DIR_BACKEND, 'checkpoint_numbers')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        verbose =2,
        mode='min',
        save_best_only=True)

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    #               loss="categorical_crossentropy",
    #               metrics=['accuracy'])

    # model.load_weights(checkpoint_filepath)
    model.fit(x = train_dataset, epochs=1000, callbacks=[model_checkpoint_callback])

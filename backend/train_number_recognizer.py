from PIL import ImageFont, ImageDraw, Image
from os.path import abspath, join, dirname
import os

from tensorflow._api.v2 import data

from common import DIR_SIGN_DATA
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

def main():    
    DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    model_path = join(DIR_BACKEND, 'models', 'checkpoint_sign_detection')
    model_path_finetuned = join(DIR_BACKEND, 'models', 'checkpoint_sign_detection_finetuned')
    data_dir = DIR_SIGN_DATA
    train_model(data_dir, model_path)
    # finetune_model(model_path, data_dir, model_path_finetuned)


def get_number_model(num_classes, trainable_layers=None, optimizer=tf.keras.optimizers.Adam(0.001)):
    input_shape = (128, 128, 3)
    # value_range = (20, 135)
    # num_classes = value_range[1] - value_range[0] + 1

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

    base_model.trainable = trainable_layers != None
    
    if trainable_layers != None:
        fine_tune_at = len(base_model.layers) - trainable_layers
        base_model.trainable = True
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base_model.trainable = False
  
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training = False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    # x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    # x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
    # x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
    x=tf.keras.layers.Dropout(0.5)(x)
    preds=tf.keras.layers.Dense(num_classes,activation='softmax')(x) #final layer with softmax activation
    model = tf.keras.Model(inputs, preds)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

# def prep_image_to_grayscale(rgb):
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     img = np.repeat(np.expand_dims(gray, -1), 3, -1)    
#     img = img.astype(np.float32) / 255.0
#     img = (img - 0.5) * 2
#     return img    

def prep_image_training(x):   
    x /= 127.5
    x -= 1.
    return x
    # prepped_image = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    # return prepped_image 

def get_number_dataset_train(data_dir, seed=42, subset='training'):
    datagen = ImageDataGenerator(
        preprocessing_function=prep_image_training,
        featurewise_center=False,
        featurewise_std_normalization=False,
        # rotation_range=3,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        horizontal_flip=False,
        # zoom_range=(0.8, 1.2),
        validation_split=0.2)

    train_dataset = datagen.flow_from_directory(data_dir, target_size=(128, 128), color_mode = 'rgb', class_mode='categorical', batch_size = 128, seed=seed, subset=subset)
    return train_dataset

def get_number_dataset_validation(data_dir, seed=42, batch_size=128, subset='validation'):
    datagen = ImageDataGenerator(preprocessing_function=prep_image_training, validation_split=0.2)

    train_dataset = datagen.flow_from_directory(data_dir, target_size=(128, 128), color_mode = 'rgb', class_mode='categorical', batch_size = batch_size, seed=seed, subset=subset)
    return train_dataset

def train_model(data_dir, output_path):           
    seed = 42
    train_dataset = get_number_dataset_train(data_dir, seed)
    validation_dataset = get_number_dataset_validation(data_dir, seed)    
    model = get_number_model(len(train_dataset.class_indices))

    # base_learning_rate = 0.001
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
        save_weights_only=True,
        monitor='val_loss',
        verbose =2,
        mode='min',
        save_best_only=True)

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    #               loss="categorical_crossentropy",
    #               metrics=['accuracy'])

    # model.load_weights(checkpoint_filepath)
    # model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"])
    model.fit(x = train_dataset, validation_data=validation_dataset, epochs=1000, callbacks=[model_checkpoint_callback])


def finetune_model(model_path, data_dir, output_path):
    seed = 42
    train_dataset = get_number_dataset_train(data_dir, seed)
    validation_dataset = get_number_dataset_validation(data_dir, seed)
    model = get_number_model(num_classes=len(train_dataset.class_indices), trainable_layers=200, optimizer=tf.keras.optimizers.Adam(1e-5))
    model.load_weights(model_path)


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            save_weights_only=True,
            monitor='val_loss',
            verbose =2,
            mode='min',
            save_best_only=True)

    model.fit(x = train_dataset, validation_data=validation_dataset, epochs=1000, callbacks=[model_checkpoint_callback])


if __name__ == '__main__':
    main()

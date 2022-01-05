from PIL import ImageFont, ImageDraw, Image
import os
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

from numpy.lib.utils import info

from os.path import abspath, join, dirname
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))


input_shape = (70, 35, 3)
value_range = (5, 150)
num_classes = value_range[1] - value_range[0] + 1


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

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=input_shape)
x = preprocess_input(inputs)
x = base_model(x)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
preds=tf.keras.layers.Dense(num_classes,activation='softmax')(x) #final layer with softmax activation
model = tf.keras.Model(inputs, preds)

checkpoint_filepath = join(DIR_BACKEND, 'checkpoints')
model.load_weights(checkpoint_filepath)

img = Image.open('test_image.png').convert('L')
img = np.repeat(np.expand_dims(np.transpose(np.array(img)), -1), 3, -1)
img = img / (255 * 0.5) - 1
q = model.predict( np.array( [img,] )  )

cv2.imshow('wnd', img)
cv2.waitKey()
print(22)

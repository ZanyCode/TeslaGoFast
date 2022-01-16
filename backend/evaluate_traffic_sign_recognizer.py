from os.path import abspath, join, dirname
from keras_preprocessing.image import directory_iterator
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from train_traffic_sign_recognizer import get_traffic_sign_model, get_traffic_sign_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

def get_class_name(class_indices, label):
    return [key for key, value in class_indices.items() if value == label][0]

def save_as_tflite(model):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
    tflite_model = converter.convert()
    # Save the model.

    with open(join(DIR_BACKEND, 'traffic_signs.tflite'), 'wb') as f:
        f.write(tflite_model)

def get_traffic_sign_evaluation_dataset() -> directory_iterator:
    def prep_fn(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = np.repeat(np.expand_dims(gray, -1), 3, -1)        
        return gray

    datagen = ImageDataGenerator(
        preprocessing_function=prep_fn,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        zoom_range=0,
        validation_split=0.2)

    train_dataset = datagen.flow_from_directory(join(DIR_BACKEND, 'evaluation', 'max_speed'), target_size=(128, 128), color_mode = 'rgb', class_mode='categorical', batch_size = 128)        
    return train_dataset

if __name__ == '__main__':
    checkpoint_filepath = join(DIR_BACKEND, 'checkpoints_traffic_signs')
    train_dataset = get_traffic_sign_dataset()
    eval_datset = get_traffic_sign_evaluation_dataset()
    images, actual_labels = eval_datset.next()
    model = get_traffic_sign_model()
    model.load_weights(checkpoint_filepath) 

    # save_as_tflite(model)
    
    # im = Image.open(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png')).convert('L')
    # im = im.rotate(180)
    # im = im.crop((305, 240, 433, 368))
    # im_array = np.repeat(np.expand_dims(np.array(im), -1), 3, -1).astype(np.float32)
    # predicted_label = np.argmax(model.predict(np.array([im_array]))[0])
    # predicted_class = get_class_name(eval_datset.class_indices, predicted_label)
    # print(predicted_class)
    # cv2.imshow('wnd', im_array.astype(np.uint8))
    # cv2.waitKey()

    for image, actual_onehot_label in zip(images, actual_labels):
        predicted_label = np.argmax(model.predict(np.array([image]))[0])
        actual_label = np.argmax(actual_onehot_label)
        predicted_class = get_class_name(train_dataset.class_indices, predicted_label)
        actual_class = get_class_name(train_dataset.class_indices, actual_label)
        print(f'Acutal: {actual_class}, Predicted: {predicted_class}')

        cv2.imshow('wnd', image.astype(np.uint8))
        cv2.waitKey()
    # model.evaluate(x = eval_datset, steps = 10)

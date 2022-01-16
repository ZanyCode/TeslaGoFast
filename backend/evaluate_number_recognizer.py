from os.path import abspath, join, dirname
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from train_number_recognizer import get_number_dataset_validation, get_number_model
from train_traffic_sign_recognizer import get_traffic_sign_model, get_traffic_sign_dataset
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
# checkpoint_filepath = join(DIR_BACKEND, 'checkpoint_numbers_finetuned')
checkpoint_filepath = join(DIR_BACKEND, 'checkpoint_numbers')
data_dir_current_speed = join(DIR_BACKEND, 'evaluation', 'current_speed')
data_dir_max_speed = join(DIR_BACKEND, 'evaluation', 'max_speed')


def get_class_name(class_indices, label):
    return [key for key, value in class_indices.items() if value == label][0]

def evaluate_visually(data_dir):
    eval_datset = get_number_dataset_validation(data_dir, subset=None)
    images, actual_labels = next(eval_datset)
    model = get_number_model()
    model.load_weights(checkpoint_filepath)

    for image, actual_onehot_label in zip(images, actual_labels):
        predicted_label = np.argmax(model.predict(np.array([image]))[0])
        actual_label = np.argmax(actual_onehot_label)
        predicted_class = get_class_name(eval_datset.class_indices, predicted_label)
        actual_class = get_class_name(eval_datset.class_indices, actual_label)
        print(f'Acutal: {actual_class}, Predicted: {predicted_class}')

        cv2.imshow('wnd', ((image + 1) * 127.5).astype(np.uint8))
        cv2.waitKey()

def evaluate_with_real_image():
    eval_datset = get_number_dataset_validation(data_dir_current_speed)
    model = get_number_model()
    model.load_weights(checkpoint_filepath)

    im = Image.open(join(DIR_BACKEND, '2021-12-29_09-41-01_snapshot.png')).convert('L')
    im = im.rotate(180)
    im = im.crop((185, 255, 304, 322))
    ima = np.array(im)
    ima = ima.astype(np.float32) / 255.0
    ima = (ima - 0.5) * 2

    im_array = np.repeat(np.expand_dims(np.array(ima), -1), 3, -1).astype(np.float32)
    predicted_label = np.argmax(model.predict(np.array([im_array]))[0])
    predicted_class = get_class_name(eval_datset.class_indices, predicted_label)
    print(predicted_class)

    cv2.imshow('wnd', ((im_array + 1) * 127.5).astype(np.uint8))
    cv2.waitKey()


def evaluate_with_dataset(dataset_dir, model_path):
    eval_dataset = get_number_dataset_validation(dataset_dir)
    model = get_number_model()
    model.load_weights(model_path)
    model.evaluate(eval_dataset)


if __name__ == '__main__':
    evaluation_data_dir = join(DIR_BACKEND, 'training', 'merged')
    # evaluate_with_dataset(evaluation_data_dir, checkpoint_filepath)
    evaluate_visually(evaluation_data_dir)
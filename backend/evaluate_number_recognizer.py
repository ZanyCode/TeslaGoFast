from os.path import abspath, join, dirname
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from train_number_recognizer import get_number_dataset, get_number_model
from train_traffic_sign_recognizer import get_traffic_sign_model, get_traffic_sign_dataset
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
checkpoint_filepath = join(DIR_BACKEND, 'checkpoint_numbers')
data_dir = join(DIR_BACKEND, 'train_data_numbers')


def get_class_name(class_indices, label):
    return [key for key, value in class_indices.items() if value == label][0]


def save_as_tflite():
    model = get_number_model()
    model.load_weights(checkpoint_filepath) 
    eval_datset = get_number_dataset(data_dir)
    batch_images, batch_labels = next(eval_datset)

    # A generator that provides a representative dataset
    def representative_data_gen():
        for i in range(100):
            image = np.expand_dims(batch_images[i], 0)         
            yield [image]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This sets the representative dataset for quantization
    converter.representative_dataset = representative_data_gen
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
    converter.target_spec.supported_types = [tf.int8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open('number_recognizer_67_119_quant.tflite', 'wb') as f:
        f.write(tflite_model)

def compare_models():
    eval_datset = get_number_dataset(data_dir)
    batch_images, batch_labels = next(eval_datset)

    # Quantized Tflite model
    def set_input_tensor(interpreter, input):
        input_details = interpreter.get_input_details()[0]
        tensor_index = input_details['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        # Inputs for the TFLite model must be uint8, so we quantize our input data.
        # NOTE: This step is necessary only because we're receiving input data from
        # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
        # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
        #   input_tensor[:, :] = input
        scale, zero_point = input_details['quantization']
        input_tensor[:, :] = np.uint8(input / scale + zero_point)
        # input_tensor[:, :] = np.uint8(input + 1.0 / 127.5)
        # input_tensor[:, :] = input

    def classify_image(interpreter, input):
        set_input_tensor(interpreter, input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details['index'])
        # Outputs from the TFLite model are uint8, so we dequantize the results:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
        top_1 = np.argmax(output)
        return top_1

    interpreter = tf.lite.Interpreter('number_recognizer_67_119_quant.tflite')
    interpreter.allocate_tensors()

    # Collect all inference predictions in a list
    batch_prediction = []
    batch_truth = np.argmax(batch_labels, axis=1)

    for i in range(len(batch_images)):
        prediction = classify_image(interpreter, batch_images[i])
        batch_prediction.append(prediction)

    # Compare all predictions to the ground truth
    tflite_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy(batch_prediction, batch_truth)
    print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))

    # Raw Model
    model = get_number_model()
    model.load_weights(checkpoint_filepath)     
    logits = model(batch_images)
    prediction = np.argmax(logits, axis=1)
    truth = np.argmax(batch_labels, axis=1)

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))


def evaluate_visually():
    eval_datset = get_number_dataset(data_dir)
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
    eval_datset = get_number_dataset(data_dir)
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


if __name__ == '__main__':   
    # save_as_tflite()
    compare_models()
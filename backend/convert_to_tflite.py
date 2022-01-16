import tensorflow as tf
import numpy as np
import os
import cv2
from os.path import abspath, join, dirname
from train_number_recognizer import get_number_dataset_train, get_number_dataset_validation, get_number_model
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu


def main():
    DIR_BACKEND = abspath(join(dirname(abspath(__file__))))
    # raw_model_path = join(DIR_BACKEND, 'checkpoint_numbers_finetuned')
    raw_model_path = join(DIR_BACKEND, 'checkpoint_numbers')
    representative_dataset_dir = join(DIR_BACKEND, 'training', 'merged')
    tflite_model_path = join(DIR_BACKEND, 'tgf_quant.tflite')
    # save_as_tflite(raw_model_path, representative_dataset_dir, tflite_model_path)
    compare_models(raw_model_path, tflite_model_path, representative_dataset_dir)


def get_class_name(class_indices, label):
    return [key for key, value in class_indices.items() if value == label][0]


def save_as_tflite(checkpoint_filepath, representative_dataset_dir, output_filename):
    model = get_number_model()
    model.load_weights(checkpoint_filepath)
    # representative_dataset = get_number_dataset_validation(representative_dataset_dir)
    batch_size = 100
    representative_dataset = get_number_dataset_validation(representative_dataset_dir, batch_size=batch_size)
    batch_images, batch_labels = next(representative_dataset)

    # A generator that provides a representative dataset
    def representative_data_gen():
        for i in range(batch_size):
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

    with open(output_filename, 'wb') as f:
        f.write(tflite_model)


def compare_models(raw_model_path, tflite_model_path, eval_dataset_path):    
    eval_datset = get_number_dataset_validation(eval_dataset_path)
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

    interpreter = tf.lite.Interpreter(tflite_model_path)
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
    model.load_weights(raw_model_path)
    logits = model(batch_images)
    prediction = np.argmax(logits, axis=1)
    truth = np.argmax(batch_labels, axis=1)

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))


if __name__ == '__main__':
    main()
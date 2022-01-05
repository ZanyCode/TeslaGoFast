import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
from os.path import abspath, join, dirname
DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

path_to_frozen_graphdef_pb = join(DIR_BACKEND, 'resnet_traffic_signs', 'inference_graph', 'frozen_inference_graph.pb')

gf = tf.compat.v1.GraphDef()
m_file = open(path_to_frozen_graphdef_pb,'rb')
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('somefile.txt','r')
data = file.readlines()
print("output name = ")
print(data[len(data)-1])

print("Input name = ")
file.seek ( 0 )
print(file.readline())

# Converting a GraphDef from file.
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  path_to_frozen_graphdef_pb, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# input_tensors = [...]
# output_tensors = [...]
# frozen_graph_def = tf.GraphDef()
# with open(path_to_frozen_graphdef_pb, 'rb') as f:
#   frozen_graph_def.ParseFromString(f.read())
# tflite_model = tf.contrib.lite.toco_convert(frozen_graph_def, input_tensors, output_tensors)


# interpreter = tf.lite.Interpreter(join(DIR_BACKEND, 'gtsrb_model.lite'))
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(33)

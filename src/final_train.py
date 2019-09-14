import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', 'human_detection_graph.pbtxt')

    saver.save(K.get_session(), 'out/human_detection.chkp')

    freeze_graph.freeze_graph('out/human_detection_graph.pbtxt', None, False, 'out/human_detection.chkp', output_node_name, "save/restore_all", "save/Const:0", 'out/frozen_human_detection.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_human_detection.pb', "rb") as f: input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names, [output_node_name], tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_human_detection.pb', "wb") as f: f.write(output_graph_def.SerializeToString())

IMG_WIDTH, IMG_HEIGHT = 100, 100

EPOCHS = 50
BATCH_SIZE = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('data/train', target_size = (IMG_WIDTH, IMG_HEIGHT), batch_size = BATCH_SIZE, color_mode = "rgb", class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory('data/test', target_size = (IMG_WIDTH, IMG_HEIGHT), batch_size = BATCH_SIZE, color_mode = "rgb", class_mode = 'categorical')
model.fit_generator(train_generator, steps_per_epoch=1832 // BATCH_SIZE, epochs = EPOCHS, validation_data = validation_generator, validation_steps = 741 // BATCH_SIZE)

model.save_weights('weights.h5')

export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Sigmoid")
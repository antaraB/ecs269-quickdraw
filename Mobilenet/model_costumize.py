import keras
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
#import matplotlib.pyplot as plt
import os
import numpy as np

from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D

import h5py

f1 = h5py.File('data/x_test.h5', 'r')
f2 = h5py.File('data/x_train.h5', 'r')
f3 = h5py.File('data/y_test.h5', 'r')
f4 = h5py.File('data/y_train.h5', 'r')

X1 = f1['name-of-dataset']
test_images= np.array(X1.value)

X2 = f2['name-of-dataset']
train_images= np.array(X2.value)

X3 = f3['name-of-dataset']
test_labels= np.array(X3.value)

X4 = f4['name-of-dataset']
train_labels= np.array(X4.value)

print("train_images")
print(train_images.shape)

print("test_images")
print(test_images.shape)


print("train_images", train_images.shape)
train_images = np.reshape(train_images, (10000,28,28))

test_images = np.reshape(test_images, (2000,28,28))

print("train_images", train_images.shape)

train_images = train_images.astype('float32') / 255.0

test_images = test_images.astype('float32') / 255.0


from keras.utils import to_categorical
#encoded_y_train = to_categorical(train_labels, num_classes=20, dtype='float32')
encoded_y_train = to_categorical(train_labels, num_classes=20)
encoded_y_test = to_categorical(test_labels, num_classes=20)

target_size = 128
from skimage.transform import resize

def preprocess_image(x):
    x = resize(x, (target_size, target_size),
            mode='constant',
            anti_aliasing=False)

    x = np.stack((x,)*3, axis=-1)

    return x.astype(np.float32)


from sklearn.utils import shuffle
def load_data_generator(x, y, batch_size=64):
    num_samples = x.shape[0]
    while 1:  # Loop forever so the generator never terminates
        try:
            shuffle(x)
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im) for im in x[i:i+batch_size]]
                y_data = y[i:i + batch_size]

                # convert to numpy array since this what keras required
                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            print(err)

from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model

def build_model():
    shallow=True
    input_tensor=None
    input_shape=None
    classes = 20
    input_tensor = Input(shape=(target_size, target_size, 3))
    alpha=1
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)



    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='mobilenet')
    return model


model1 = build_model()

from keras.optimizers import Adam
model1.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


train_generator = load_data_generator(train_images, encoded_y_train, batch_size=8)

print(train_images.shape)
print(train_labels.shape)

model1.fit_generator(
    generator=train_generator,
    steps_per_epoch=1250,
    verbose=1,
    epochs=5)


test_generator = load_data_generator(test_images, encoded_y_test, batch_size=8)
test_loss, test_acc = model1.evaluate_generator(generator=test_generator,
                         steps=200,
                         verbose=1)

print('Test accuracy:', test_acc)

model_name = "tf_serving_keras_mobilenetv2"
model1.save("models/{}.h5".format(model_name))

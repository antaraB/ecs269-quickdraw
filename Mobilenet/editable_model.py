# from tensorflow import keras
import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications import MobileNet
from keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D, Convolution2D, DepthwiseConv2D, BatchNormalization, Activation
from keras.models import Sequential, Model
# import matplotlib.pyplot as plt
import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')           # noqa: E402
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import tensorflow as tf

from keras.utils import to_categorical
from skimage.transform import resize
import h5py

from keras.optimizers import Adam
from sklearn.utils import shuffle
 
#STEP 1 : Create a classification model with accuracy of above 90%
#import the fashion mnist data

#path = 'data_10perclass'
path ='data'
f1 = h5py.File(path+'/x_test.h5', 'r')
f2 = h5py.File(path+'/x_train.h5', 'r')
f3 = h5py.File(path+'/y_test.h5', 'r')
f4 = h5py.File(path+'/y_train.h5', 'r')

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


#encoded_y_train = to_categorical(train_labels, num_classes=20, dtype='float32')
#encoded_y_test = to_categorical(test_labels, num_classes=20, dtype='float32')

encoded_y_train = to_categorical(train_labels, num_classes=20)
encoded_y_test = to_categorical(test_labels, num_classes=20)
target_size = 224


def preprocess_image(x):
    # Resize the image to have the shape of (96,96)
    x = resize(x, (target_size, target_size),
            mode='constant',
            anti_aliasing=False)
    
    # convert to 3 channel (RGB)
    x = np.stack((x,)*3, axis=-1) 
    
    # Make sure it is a float32, here is why 
    # https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow
    return x.astype(np.float32)


def load_data_generator(x, y, batch_size=20):
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


def build_model(shallow=False ):
    alpha = 1
    classes = 20
    input_tensor = Input(shape=(target_size, target_size, 3))

    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

################2
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

#######################3
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
##############################4
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#########################5
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
##########################6
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
######################### 7
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
###################### 8
    if not shallow:
        for _ in range(5):
            x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
#########################9
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
##################10
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
################## 11
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    # base_model = MobileNetV2(
    #     include_top=False,
    #     weights='imagenet',
    #     input_tensor=input_tensor,
    #     input_shape=(target_size, target_size, 3),
    #     pooling='avg')

    # for layer in base_model.layers:
    #     layer.trainable = True  # trainable has to be false in order to freeze the layers
        
    # op = Dense(256, activation='relu')(base_model.output)
    # op = Dropout(.25)(op)

    # # softmax: calculates a probability for every possible class.
    # #
    # # activation='softmax': return the highest probability;
    # # for example, if 'Coat' is the highest probability then the result would be 
    # # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    # ##
    # output_tensor = Dense(20, activation='softmax')(op)



    return model

model1 = build_model()

model1.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


train_generator = load_data_generator(train_images, encoded_y_train, batch_size=20)

print(train_images.shape)
print(train_labels.shape)

model1.fit_generator(
    generator=train_generator,
    steps_per_epoch=500,
    verbose=1,
    epochs=5)

test_generator = load_data_generator(test_images, encoded_y_test, batch_size=8)
test_loss, test_acc = model1.evaluate_generator(generator=test_generator,
                         steps=125,
                         verbose=1)

print('Test accuracy:', test_acc)

model_name = "e_500_5_mobilenetv2"
model1.save("models/{}.h5".format(model_name))


from tensorflow import keras
import keras
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
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
encoded_y_train = to_categorical(train_labels, num_classes=20, dtype='float32')
encoded_y_test = to_categorical(test_labels, num_classes=20, dtype='float32')

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

def build_model( ):
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
        
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)
    
    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be 
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(20, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)


    return model

model1 = build_model()

from keras.optimizers import Adam
model1.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


train_generator = load_data_generator(train_images, encoded_y_train, batch_size=64)

print(train_images.shape)
print(train_labels.shape)

model1.fit_generator(
    generator=train_generator,
    steps_per_epoch=157,
    verbose=1,
    epochs=8)


test_generator = load_data_generator(test_images, encoded_y_test, batch_size=64)
test_loss, test_acc = model1.evaluate_generator(generator=test_generator,
                         steps=900,
                         verbose=1)

print('Test accuracy:', test_acc)

model_name = "tf_serving_keras_mobilenetv2"
model1.save("models/{}.h5".format(model_name))

#Predicting an example image
# predictions = model1.predict(test_images)

# print (predictions[0])

# print (np.argmax(predictions[0]))

# print (test_labels[0])

# # Grab an image from the test dataset
# img = test_images[0]

# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img,0))

# print(img.shape)

# predictions_single = model1.predict(img)

# print(predictions_single)

# print (np.argmax(predictions_single[0]))

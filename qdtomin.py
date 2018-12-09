##Take quickdraw npy files and convert them into a test/train MNIST like dataset

import numpy as np
from sklearn.model_selection import train_test_split
from os import walk, getcwd
import h5py


##Place all the npy quickdraw files here:
mypath = "data/"
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    if filenames != '.DS_Store':       ##Ugh mac junk
        txt_name_list.extend(filenames)
        break


x_train = []
x_test = []
x_test_1 = []
y_train = []
y_test = []
y_test_1 = []
xtotal = []
ytotal = []
slice_train = int(2000/len(txt_name_list))  ###Setting value to be 80000 for the final dataset
print("slice_train", slice_train)
i = 0
seed = np.random.randint(1, 10e6)


# ##Creates test/train split with quickdraw data
# for txt_name in txt_name_list:
#     txt_path = mypath + txt_name
#     x = np.load(txt_path)
#     x = x.astype('float32') / 255.    ##scale images
#     y = [i] * len(x)  
#     # np.random.seed(seed)
#     # np.random.shuffle(x)
#     # np.random.seed(seed)
#     # np.random.shuffle(y)
#     x = x[:slice_train]
#     y = y[:slice_train]
#     if i != 0: 
#         xtotal = np.concatenate((x,xtotal), axis=0)
#         ytotal = np.concatenate((y,ytotal), axis=0)
#     else:
#         xtotal = x
#         ytotal = y
#     i += 1
# x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, shuffle=False)


for txt_name in txt_name_list:
    txt_path = mypath + txt_name
    print("txt_name : ",txt_name)
    print("label : ",i)
    x = np.load(txt_path)
    x = x.astype('float32') / 255.    ##scale images
    y = [i] * len(x)  
    # np.random.seed(seed)
    # np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    x = x[:1100]
    y = y[:1100]
    xt = x[-100:]
    yt = y[-100:]
    x = x[:1000]
    y = y[:1000]
    if i != 0: 
        x_train = np.concatenate((x,x_train), axis=0)
        y_train = np.concatenate((y,y_train), axis=0)
        x_test = np.concatenate((xt,x_test), axis=0)
        y_test = np.concatenate((yt,y_test), axis=0)
    else:
        x_train = x
        y_train = y
        x_test = xt
        y_test = yt
    i += 1
    # np.random.seed(seed)
    # np.random.shuffle(x_train)
    # np.random.seed(seed)
    # np.random.shuffle(y_train)
print("x_train.shape",x_train.shape)
print("y_train.shape",y_train.shape)
print("x_test.shape",x_test.shape)
print("y_test.shape",y_test.shape)


x_train, x_test_1, y_train, y_test_1 = train_test_split(x_train, y_train, test_size=0, random_state=42)


print("x_test_1.shape",x_test_1.shape)
print("y_test_1.shape",y_test_1.shape)
print("x_train.shape",x_train.shape)
print("y_train.shape",y_train.shape)
print("x_test.shape",x_test.shape)
print("y_test.shape",y_test.shape)


##Saves this out as hdf5 format
data_to_write = x_test
with h5py.File('x_test.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)
data_to_write = x_train
with h5py.File('x_train.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)
data_to_write = y_test
with h5py.File('y_test.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)
data_to_write = y_train
with h5py.File('y_train.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)

print("y train first 10")
print(y_train[:20])

print("y test first 10")
print(y_test[:10])

print("x_train")
print(x_train)

##Visualize a quickdraw file
import matplotlib.pyplot as plt
face1 = x_test[0].reshape(28,28)
print(y_test[0])
plt.imshow(face1)
plt.show()


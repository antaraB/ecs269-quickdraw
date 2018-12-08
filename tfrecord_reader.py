import tensorflow as tf
import numpy as np

i=0
a = np.zeros(20,dtype=int) #array showing number of instances in each category

for example in tf.python_io.tf_record_iterator("C:/Users/molly/QuickDraw_Tutorial/output/training.tfrecord-00000-of-00001"):

    feature_to_type = {
        "ink": tf.VarLenFeature(dtype=tf.float32),
        "shape": tf.FixedLenFeature([2], dtype=tf.int64),
        "class_index": tf.FixedLenFeature([1], dtype=tf.int64)
    }
    parsed_features = tf.parse_single_example(example, feature_to_type)

    x = tf.Session().run(parsed_features["class_index"])
    a[x]+=1
    i+=1

    #print TF record attributes 
    print(tf.Session().run(parsed_features["class_index"]))
    print(tf.Session().run(parsed_features["ink"].values))
    print(tf.Session().run(parsed_features["shape"]))

    if i==10: #number of TF records to check
        break

#print(a) #print class_index array
#print(i) #print number of TF records parsed


    


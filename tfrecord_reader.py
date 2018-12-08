import argparse
import tensorflow as tf
import numpy as np


def main(args):
    i=0
    a = np.zeros(20,dtype=int) #array showing number of instances in each category
    feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "class_index": tf.FixedLenFeature([1], dtype=tf.int64)
        }
    # for example in tf.python_io.tf_record_iterator("C:/Users/molly/QuickDraw_Tutorial/output/training.tfrecord-00000-of-00001"):
    for idx,example in enumerate(tf.python_io.tf_record_iterator(args.get('filename'))):
        
        parsed_features = tf.parse_single_example(example, feature_to_type)

        x = tf.Session().run(parsed_features["class_index"])
        a[x]+=1

        #print TF record attributes 
        print("class_index {}".format(tf.Session().run(parsed_features["class_index"])))
        # print("ink {}".format(tf.Session().run(parsed_features["ink"].values)))
        # print("shape {}".format(tf.Session().run(parsed_features["shape"])))

        if idx==10: #number of TF records to check
            break

        print("a: {}".format(a)) #print class_index array
        print("idx is {}".format(idx)) #print number of TF records parsed

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('filename',action='store', help="Complete path of json file")
    args = parser.parse_args()
    main(vars(args))
    


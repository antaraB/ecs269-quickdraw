#creates h5 files from ndjson files
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from os import walk, getcwd
import h5py

def draw_it(raw_strokes):
    image = Image.new("P", (255,255), color=255)
    image_draw = ImageDraw.Draw(image)
    for idx,stroke in enumerate(raw_strokes):
        color=(idx*2)%255
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=color, width=6)
    return np.array(image)

def parse_line(ndjson_line):
  """Parse an ndjson line and return ink (as np array)."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  inkarray = sample["drawing"]
  if not inkarray:
    print("Empty inkarray")
    return None, None

  return inkarray, class_name


def convert_data(trainingdata_dir,
                 observations_per_class,
                 output_file,
                 classnames,
                 output_shards=10,
                 offset=0,
                 mode=0):
  file_handles = []
  # Open all input files.
  for filename in sorted(tf.gfile.ListDirectory(trainingdata_dir)):
    if not filename.endswith(".ndjson"):
      print("Skipping", filename)
      continue
    file_handles.append(
        tf.gfile.GFile(os.path.join(trainingdata_dir, filename), "r"))
    if offset:  # Fast forward all files to skip the offset.
      count = 0
      for _ in file_handles[-1]:
        count += 1
        if count == offset:
          break

  reading_order = list(range(len(file_handles))) * observations_per_class
  random.shuffle(reading_order)
  images_array=[]
  labels_array=[]
  for index,c in enumerate(reading_order):
    line = file_handles[c].readline()
    ink = None
    while ink is None:
        ink, class_name = parse_line(line)
        if ink is None:
            print ("Couldn't parse ink from '" + line + "'.")
        np_array = draw_it(ink)
    if index !=0: images_array = np.concatenate((np_array,images_array),axis=0)
    else: images_array = np_array

    if class_name not in classnames:
      classnames.append(class_name)
    np_array = [classnames.index(class_name)]
    if index !=0: labels_array = np.concatenate((np_array,labels_array),axis=0)
    else: labels_array = np_array
  # generate h5 output files
  if mode==0: #training
    print("making training data output")
    data_to_write = images_array
    with h5py.File('images_train.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=data_to_write)
    data_to_write = labels_array
    with h5py.File('labels_train.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=data_to_write)
  else: #eval
    print("making eval data output")
    data_to_write = images_array
    with h5py.File('images_eval.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=data_to_write)
    data_to_write = labels_array
    with h5py.File('labels_eval.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=data_to_write)
  print(labels_array)
  # Close all files
  for f in file_handles:
    f.close()
  return classnames


def main(argv):
  del argv
  classnames = convert_data(
      FLAGS.ndjson_path,
      FLAGS.train_observations_per_class,
      os.path.join(FLAGS.output_path, "training.tfrecord"),
      classnames=[],
      offset=0,
      mode=0)
  convert_data(
      FLAGS.ndjson_path,
      FLAGS.eval_observations_per_class,
      os.path.join(FLAGS.output_path, "eval.tfrecord"),
      classnames=classnames,
      offset=FLAGS.train_observations_per_class,
      mode=1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--ndjson_path",
      type=str,
      default="ndjson_files",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="output",
      help="Directory where to store the output TFRecord files.")
  parser.add_argument(
      "--train_observations_per_class",
      type=int,
      default=1000,
      help="How many items per class to load for training.")
  parser.add_argument(
      "--eval_observations_per_class",
      type=int,
      default=100,
      help="How many items per class to load for evaluation.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

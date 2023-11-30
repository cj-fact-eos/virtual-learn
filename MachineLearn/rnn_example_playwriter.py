import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import numpy as np
import tensorflow as tf

path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print(f"Length of file : {len(text)} characters")
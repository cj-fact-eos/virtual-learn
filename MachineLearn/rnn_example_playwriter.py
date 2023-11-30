import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import numpy as np
import tensorflow as tf
 
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

# Get the length of the character in the file
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print(f"Length of file : {len(text)} characters")

# Print First 300 Character
print(text[:300])

# Get how many unique character are present in the code.
vocab = sorted(set(text))
print(f"{len(vocab)} unique character")

# print the unique characters.
uniqueCharacters = ""
for i in vocab:
    uniqueCharacters += i

# they are alphabets and symbols and if we try more characters than it would add number to it also
# the below printed are unique characters that Neural Network will consume as token 
print(uniqueCharacters)



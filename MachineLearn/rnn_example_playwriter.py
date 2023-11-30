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

# Vectorize the text 
# i.e. Before training, you need to convert the string to a numerical representation 
# to do so we will use tf.keras.layers.StringLookup 
# it converts each characters into a numeric ID. It just needs the text to be split into token first.

example_text = ["abcdefghijklmnop",'xyz']
chars = tf.strings.unicode_split(example_text, input_encoding="UTF-8")
print(chars)

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)

ids = ids_from_chars(chars)
print(ids)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

chars = chars_from_ids(ids)
print(chars)



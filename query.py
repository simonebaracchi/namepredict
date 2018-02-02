import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from math import ceil, inf
import operator
import h5py
import sys

def encode_word(max_len, word):
    """
    One-hot-encode a word.
    Return an array of 26*max_len, each 26 items representing a letter.
    """
    ret = [0] * max_len * 26
    count = 0
    for letter in word:
        ret[(count*26) + (ord(letter) - ord('A'))] = 1
        count += 1
    return ret

def one_hot(size, idx):
    """
    one-hot-encode a single integer value or a list
    """
    ret = [0] * size
    if type(idx) is list:
        for i in idx:
            ret[i] = 1
    else:
        ret[idx] = 1
    return ret

if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
    print('Usage: ' + str(sys.argv[0]) + ' <surname>')
    sys.exit(1)

# last names (inputs) => first names (output)
X = {}
# unique outputs (classes)
all_Y = {}
# input max length (used to reduce input size later)
max_X = 0
# load input+outputs from file. file format is "SURNAME NAME".
with open('names.txt') as f:
    total = f.readlines()
    for line in total:
        words = line.split()
        # add newly found name
        if words[1] not in all_Y:
            all_Y[words[1]] = len(all_Y)
        # add to list of names associated to last name
        if words[0] not in X:
            X[words[0]] = []
        X[words[0]].append( all_Y[words[1]] )
        max_X = max(max_X, len(words[0]))
print("Loaded data. max_X = {}, classes={}\n".format(max_X, len(all_Y)))

model = Sequential([
    Dense(len(all_Y), input_dim=max_X * 26),
    Activation('relu'),
    Dense(len(all_Y)),
    Activation('relu'),
    Dense(len(all_Y)),
    Activation('softmax'),
])

model.load_weights("model.h5")

test_out = model.predict(numpy.array([encode_word(max_X, sys.argv[1])]))
# iter results highest-first. there probably is a better way
to_order = {}
for i in range(0, len(test_out[0])):
    to_order[i] = test_out[0][i]
print("Predicted names:")
ordered = sorted(to_order.items(), key=operator.itemgetter(1), reverse=True)
for i in range(0, 10):
    # convert back to string
    predicted = [ x for x in all_Y if all_Y[x] == ordered[i][0] ][0]
    print("\t{} (score of {})".format(predicted, ordered[i][1]))



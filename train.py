import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from math import ceil
import operator
import h5py

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
    one-hot-encode a single integer value
    """
    ret = [0] * size
    ret[idx] = 1
    return ret


seed = 7
numpy.random.seed(seed)

# last names (inputs)
X = []
# first names (outputs)
Y = []
# unique outputs (classes)
all_Y = {}
# input max length (used to reduce input size later)
max_X = 0
# load input+outputs from file. file format is "SURNAME NAME".
with open('nomi.txt') as f:
    total = f.readlines()
    for line in total:
        words = line.split()
        X.append(words[0])
        Y.append(words[1])
        max_X = max(max_X, len(words[0]))
        if words[1] not in all_Y:
            all_Y[words[1]] = len(all_Y)
print("Loaded data. max_X = {}, classes={}\n".format(max_X, len(all_Y)))

model = Sequential([
    Dense(len(all_Y), input_dim=max_X * 26),
    Activation('relu'),
    Dense(len(all_Y)),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

def generate_training_set(model, stop):
    for i in range(0, min(stop, len(X))):
        # stop sooner to allow debugging ...
        if i == stop:
            break
        if i > 0 and 0 == i % 5000:
            print("Batch {} of {}...".format(i, len(X)))
            model.save("nomi.h5")
        enc_in = numpy.array([encode_word(max_X, X[i])])
        enc_out = numpy.array([one_hot(len(all_Y), all_Y[Y[i]])])
        yield({'dense_1_input': enc_in}, {'activation_2': enc_out})

# train 
stop_at = 100
model.fit_generator(generate_training_set(model, stop_at), steps_per_epoch=min(stop_at, len(X)), epochs=1 if stop_at else 10)

test_out = model.predict(numpy.array([encode_word(max_X, "PAVAROTTI")]))
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



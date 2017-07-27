import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(series.shape[0] - window_size):
        X.append(series[i:i + window_size])

    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data


def build_part1_RNN(window_size):
    # as the first layer in a Sequential model
    batch_size = 2

    model = Sequential()

    model.add(LSTM(5, input_shape=(window_size, 1)))

    # model.add(LSTM(5, batch_input_shape=(batch_size, window_size, 1), return_sequences=True))
    # model.add(LSTM(5, batch_input_shape=(batch_size, window_size, 1)))

    model.add(Dense(1))
    return model

# TODO: return the text input with only ascii lowercase and the
# punctuation given below included.


def cleaned_text(text):

    # I have added a space and a hyphen here
    punctuation = ['!', ',', '.', ':', ';', '?', ' ', '-']

    # we can have punctuation or lowercase letters
    text = [c for c in text.lower() if c in punctuation or c.isalpha()]

    return "".join(text)

# TODO: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    target = len(text) - window_size
    i = 0
    while i < target:
        i_plus_window_size = i + window_size
        inputs.append(text[i:i_plus_window_size])
        outputs.append(text[i_plus_window_size:i_plus_window_size + 1])
        i += step_size

    return inputs, outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation,
# categorical_crossentropy loss


def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation(activation='softmax'))
    return model


"""
Train our model
"""

import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score


def main():
    # Grab out training data
    df = pd.read_csv('./data/training.txt',
                     sep='\t',
                     header=None,
                     dtype=np.int32)

    # Pull out features and labels. We'll want out labels to be one-hot encoded
    # for easier training.
    y = df.ix[:, 0]
    X = df.ix[:, 1:]
    y_hot = to_categorical(y, num_classes=4)

    # In no way optimized
    model = Sequential()
    model.add(Dense(24, activation='relu', input_dim=32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Time to hit the gym
    model.fit(X, y_hot, epochs=1000, batch_size=64)

    # Save our model for validation later
    model.save('./output/model.h5')


if __name__ == '__main__':
    main()

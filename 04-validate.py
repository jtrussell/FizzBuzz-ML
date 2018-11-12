"""
Get some metrics about our model's performance
"""

import itertools
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix for our model.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.title('Confusion Matrix - It\'s All Confusing')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True Fizzin and Buzzin')
    plt.xlabel('Predicted Fizzin and Buzzin')
    plt.tight_layout()
    plt.savefig('./output/confusion-matrix.png')
    plt.close()


def main():
    # Unpack our saved model
    model = load_model('./output/model.h5')

    # Get testing data and labels, note that these were *not* used during
    # training.
    df = pd.read_csv('./data/testing.txt',
                     sep='\t',
                     header=None,
                     dtype=np.int32)
    y = df.ix[:, 0]
    X = df.ix[:, 1:]

    # Predict labels for our test data. These will be in the form of
    # probabilities for each possible label. From that we collect the single
    # label with highest probability.
    y_pred = model.predict(X)
    y_pred = [np.argmax(proba) for proba in y_pred]

    # Plot a confusion matrix
    classes = ['Nothing', 'Fizz', 'Buzz', 'FizzBuzz']
    plot_confusion_matrix(y, y_pred, classes)

    # Report our overal accuracy
    acc = accuracy_score(y, y_pred)
    print('Accuracy against test set: {}'.format(acc))


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASK_TOKEN = 'MASK_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'


def pad_vec(vec, length):
    """
    Pads a vector to a specified length
    :param vec: the original vector, must not be longer than length
    :param length: the desired length
    :return: the padded vector
    """
    r = np.zeros((1, length))
    for i, v in enumerate(vec):
        r[0][i] = v
    return r


def plot_training(path, history):
    """
    Plots loss and accuracy throughout training phase.
    :param path: the existing directory to save the plots to
    :param history: the History object created by model.fit().
    :return: None
    """
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.savefig(path + '/loss.png')
    plt.close()

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['Training set', 'Validation set'], loc='lower right')
    plt.savefig(path + '/acc.png')
    plt.close()


def precision(preds, true, label):
    """
    Calculates preision over a given label.
    :param preds: vector containing prediction labels.
    :param true: vector containing true labels.
    :param label: the label to calculate on
    :return: the label's precision.
    """
    true_pos, total_pos = 0.0, 0.0
    for i in range(len(true)):
        if preds[i] == label:
            total_pos += 1.0
            if true[i] == label:
                true_pos += 1.0
    if total_pos == 0:
        return None
    return true_pos/total_pos


def recall(preds, true, label):
    """
    Calculates recall over a given label.
    :param preds: vector containing prediction labels.
    :param true: vector containing true labels.
    :param label: the label to calculate on
    :return: the label's recall.
    """
    true_pos, total_pos = 0.0, 0.0
    for i in range(len(true)):
        if true[i] == label:
            total_pos += 1.0
            if preds[i] == label:
                true_pos += 1.0
    if total_pos == 0:
        return None
    return true_pos / total_pos


def f1_score(p, r):
    """
    Calculates the F1 score with the given precision and recall.
    :param p: Precision
    :param r: Recall
    """
    if p is None or r is None:
        return None
    if p == 0 and r == 0:
        return 0
    return 2 * p * r / (p + r)

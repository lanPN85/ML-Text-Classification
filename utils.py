import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASK_TOKEN = 'MASK_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'


def pad_vec(vec, length):
    r = np.zeros((1, length))
    for i, v in enumerate(vec):
        r[0][i] = v
    return r


def plot_training(path, history, gru_lambda, dense_lambda):
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.savefig(path + '/loss_%s_%s.png' % (gru_lambda, dense_lambda))
    plt.close()

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['Training set', 'Validation set'], loc='lower right')
    plt.savefig(path + '/acc_%s_%s.png' % (gru_lambda, dense_lambda))
    plt.close()


def precision(preds, true, label):
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
    if p is None or r is None:
        return None
    if p == 0 and r == 0:
        return 0
    return 2 * p * r / (p + r)

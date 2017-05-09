# Setup proper keras environment
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

from keras.utils.np_utils import to_categorical

import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASK_TOKEN = 'MASK_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'


def batch_generator(Xt_train, Xc_train, y_train, nb_class, total_len, batch_size=5):
    while True:
        for i in range(0, total_len, batch_size):
            y = to_categorical(y_train[i:i + batch_size], nb_class)
            Xt = Xt_train[i:i + batch_size]
            Xc = Xc_train[i:i + batch_size]
            yield ([Xt, Xc], y)


def pad_vec(vec, length):
    r = np.zeros((1, length))
    for i, v in enumerate(vec):
        r[0][i] = v
    return r


def save_classifier(classifier, directory):
    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    config = {'title_output': classifier.title_output,
              'content_output': classifier.content_output,
              'dense_neurons': classifier.dense_neurons,
              'title_len': classifier.title_len,
              'content_len': classifier.content_len,
              'classes': classifier.classes,
              'word_vec_dim': np.shape(classifier.word_vec),
              'gru_reg': classifier.gru_regularize,
              'dense_reg': classifier.dense_regularize}

    classifier.model.save_weights(f1)
    pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)
    np.savez(f3, wit=classifier.word_to_index, itw=classifier.index_to_word,
             wv=classifier.word_vec)
    print('Saved model to %s.' % directory)


def load_classifier(directory, cls):
    f1 = directory + '/weights.hdf5'
    f2 = directory + '/config.pkl'
    f3 = directory + '/dictionary.npz'

    print('Loading model from %s...' % directory)
    try:
        config = pickle.load(open(f2, 'rb'))
        npz_file = np.load(f3)

        word_to_index, index_to_word, word_vec = npz_file["wit"].reshape(1)[0], npz_file["itw"], npz_file["wv"].reshape(config['word_vec_dim'])
        print('Done.')
        return cls(word_vec, word_to_index, index_to_word, config['classes'], title_output=config['title_output'],
                   content_output=config['content_output'], dense_neurons=config['dense_neurons'],
                   title_len=config['title_len'], content_len=config['content_len'], weights=f1, directory=directory,
                   gru_regularize=config.get('gru_reg', 0),
                   dense_regularize=config.get('dense_reg', 0))
    except FileNotFoundError:
        print('One or more model files cannot be found. Terminating...')
        sys.exit()


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

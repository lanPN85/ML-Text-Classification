from keras.utils.np_utils import to_categorical
from classifier import *

import pickle
import sys
import numpy as np

MASK_TOKEN = 'MASK_TOKEN'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'


def batch_generator(Xt_train, Xc_train, y_train, nb_class, total_len, batch_size=5):
    while True:
        for i in range(0, total_len, batch_size):
            y = to_categorical(y_train[i:i + batch_size], nb_class)
            Xt = Xt_train[i:i + batch_size]
            Xc = Xc_train[i:i + batch_size]
            yield ([Xt, Xc], y)


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
              'word_vec_dim': np.shape(classifier.word_vec)}

    classifier.model.save_weights(f1)
    pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)
    np.savez(f3, wit=classifier.word_to_index, itw=classifier.index_to_word,
             wv=classifier.word_vec)
    print('Saved model to %s' % directory)


def load_classifier(directory, cls=Classifier):
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
                   title_len=config['title_len'], content_len=config['content_len'], weights=f1, directory=directory)
    except FileNotFoundError:
        print('One or more model files cannot be found. Terminating...')
        sys.exit()

# Setup proper keras environment
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

from keras.layers.recurrent import GRU
from keras.layers.core import Dense
from keras.engine import merge
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input
from keras.callbacks import Callback, EarlyStopping, CSVLogger

import utils
import pickle
import nltk
import sys
import numpy as np

nltk.data.path.append('./data')


class Classifier:
    """
    Container class for a full document classifier. An instance of this class should be enough to train on and predict any two-input document.
    
    Attributes:
        word_vec: A word vector matrix for the embedding layer, where each row represents a word.
        word_to_index: A dictionary that maps each word to its corresponding position in the word_vec matrix.
        index_to_word: A list that aps each index to its word. Corresponds to word_to_index.
        classes: A list of strings that names each of the topics in the order of the dataset that the classifier's trained on.
        title_output: The number of neurons of the GRU layer that processes document title.
        content_output: The number of neurons of the GRU layer that processes document content.
        dense_neurons: The number of neurons for each of the dense layers. One layer is implicitly added on top as the output layer.
        title_len: The standardized length of documents' title.
        content_len: The standardized length of documents' content.
        weights: A file object that reads a HDF5 file which contains the saved weights of a trained model. Used for loading.
        directory: The path of the directory where the model, log files and plots should be kept.        
    """

    def __init__(self, word_vec, word_to_index, index_to_word, classes, title_output=128, content_output=512,
                 dense_neurons=(1024, 256,), title_len=50, content_len=2000, weights=None, directory='.'):
        self.directory = directory
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.title_len = title_len
        self.content_len = content_len
        self.word_vec = word_vec
        self.classes = classes
        self.title_output = title_output
        self.content_output = content_output
        self.dense_neurons = dense_neurons

        # Encode document's title
        title_inp = Input(shape=(title_len,), name='Title_Input')
        title_embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                                weights=[word_vec], mask_zero=True, name='Title_Embedding')
        self.t_encoder = Sequential(name='Title_Encoder')
        self.t_encoder.add(title_embed)
        self.t_encoder.add(GRU(title_output, name='Title_GRU', consume_less='mem'))
        title_vec = self.t_encoder(title_inp)

        # Encode document's content
        content_inp = Input(shape=(content_len,), name='Content_Input')
        content_embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                                  weights=[word_vec], mask_zero=True, name='Content_Embedding')
        self.c_encoder = Sequential(name='Content_Encoder')
        self.c_encoder.add(content_embed)
        self.c_encoder.add(GRU(content_output, name='Content_GRU', consume_less='mem'))
        content_vec = self.c_encoder(content_inp)

        # Merge vectors to create output
        doc_vec = merge(inputs=[title_vec, content_vec], mode='concat')

        # Decode using dense layers
        self.decoder = Sequential(name='Decoder')
        self.decoder.add(Dense(dense_neurons[0], input_shape=(title_output + content_output,),
                               name='Dense_0', activation='hard_sigmoid'))
        for i, n in enumerate(dense_neurons[1:]):
            self.decoder.add(Dense(n, activation='hard_sigmoid', name='Dense_%s' % (i + 1)))
        self.decoder.add(Dense(len(classes), activation='softmax', name='Dense_Output'))
        output = self.decoder(doc_vec)

        self.model = Model(input=[title_inp, content_inp], output=output, name='Model')
        if weights is not None:
            self.model.load_weights(weights)

    def log(self, string, out=True):
        """
        Utility method for logging to the model's log file and printing to stdout if needed 
        :param out: Whether there should be output to stdout 
        """
        with open(self.directory + '/log.txt', 'at') as f:
            f.write(string + '\n')
            f.close()
        if out:
            print(string)

    def compile(self, optimizer=RMSprop, learning_rate=0.0001):
        """
        Simple wrapper for compiling the model.
        :param optimizer: Function pointer to a Keras optimizer or custom optimizer function
        """
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate),
                           metrics=['accuracy'])

    def train(self, matrices, nb_epoch, batch_size=20):
        """
        Wrapper for training the classifier's model.
        :param matrices: A dictionary containing the required matrices for training.
        :return: The best validation accuracy after training and a Keras History object for plotting.
        """
        cb1 = SaveCallback(self)
        cb2 = EarlyStopping(monitor='val_loss', verbose=1, patience=4, mode='auto')
        cb3 = EarlyStopping(monitor='loss', verbose=1, patience=0, mode='auto')
        cb4 = CSVLogger(self.directory + '/epochs.csv', append=True)
        cb5 = TestCallback(self, matrices['Xt_test'], matrices['Xc_test'], matrices['y_test'])
        history = self.model.fit([matrices['Xt_train'], matrices['Xc_train']], matrices['y_train'], nb_epoch=nb_epoch,
                                 batch_size=batch_size, callbacks=[cb1, cb2, cb3, cb4, cb5], shuffle=True,
                                 validation_data=([matrices['Xt_val'], matrices['Xc_val']], matrices['y_val']))
        return history

    def predict(self, title, content, verbose=0):
        """
        Predicts the topic of a pair of document title-content.
        :param title: The document's title.
        :param content: The document's content
        :param verbose: Passed to model.predict() to decide stdout output 
        :return: The index of the predicted topic and the probability distribution of every topic.
        """
        t = nltk.word_tokenize(title.lower())
        Xt = [self.word_to_index[word] if word in self.word_to_index
              else self.word_to_index[utils.UNKNOWN_TOKEN] for word in t][:self.title_len]
        c = nltk.word_tokenize(content.lower())
        Xc = [self.word_to_index[word] if word in self.word_to_index
              else self.word_to_index[utils.UNKNOWN_TOKEN] for word in c][:self.content_len]

        Xt = utils.pad_vec(Xt, self.title_len)
        Xc = utils.pad_vec(Xc, self.content_len)

        probs = self.model.predict([Xt, Xc], verbose=verbose)
        pred = np.argmax(probs)
        return pred, probs

    def evaluate(self, Xt, Xc, y):
        """
        Evaluates the model's accuracy, precision, recall, F1 score and loss value on a dataset.
        :param Xt: Matrix containing title input.
        :param Xc: Matrix containing content input.
        :param y: Label vector.
        :return: Accuracy, lists containing each topic's precision, recall, F1 score followed by the macro-average across topics, and the model's loss value.
        """
        probs = self.model.predict([Xt, Xc], verbose=1, batch_size=100)
        preds = np.argmax(probs, axis=1)
        true = np.argmax(y, 1)
        acc = np.sum(np.equal(preds, true)) / np.size(true, 0)
        p, r, f1 = [], [], []
        for i in range(len(self.classes)):
            p.append(utils.precision(preds, true, i))
            r.append(utils.recall(preds, true, i))
            f1.append(utils.f1_score(p[i], r[i]))
        p2 = [x for x in p if x is not None]
        r2 = [x for x in r if x is not None]
        f2 = [x for x in f1 if x is not None]
        p.append(np.mean(p2))
        r.append(np.mean(r2))
        f1.append(np.mean(f2))

        print('\nCalculating loss...')
        self.compile()
        loss = self.model.evaluate([Xt, Xc], y, batch_size=100)[0]

        return acc, p, r, f1, loss

    def save(self):
        f1 = self.directory + '/weights.hdf5'
        f2 = self.directory + '/config.pkl'
        f3 = self.directory + '/dictionary.npz'

        config = {'title_output': self.title_output,
                  'content_output': self.content_output,
                  'dense_neurons': self.dense_neurons,
                  'title_len': self.title_len,
                  'content_len': self.content_len,
                  'classes': self.classes,
                  'word_vec_dim': np.shape(self.word_vec)}

        self.model.save_weights(f1)
        pickle.dump(config, open(f2, 'wb'), pickle.HIGHEST_PROTOCOL)
        np.savez(f3, wit=self.word_to_index, itw=self.index_to_word,
                 wv=self.word_vec)
        print('Saved model to %s.' % self.directory)

    @staticmethod
    def load(directory):
        f1 = directory + '/weights.hdf5'
        f2 = directory + '/config.pkl'
        f3 = directory + '/dictionary.npz'

        print('Loading model from %s...' % directory)
        try:
            config = pickle.load(open(f2, 'rb'))
            npz_file = np.load(f3)

            word_to_index, index_to_word, word_vec = npz_file["wit"].reshape(1)[0], npz_file["itw"], npz_file[
                "wv"].reshape(config['word_vec_dim'])
            print('Done.')
            return Classifier(word_vec, word_to_index, index_to_word, config['classes'],
                              title_output=config['title_output'],
                              content_output=config['content_output'], dense_neurons=config['dense_neurons'],
                              title_len=config['title_len'], content_len=config['content_len'], weights=f1,
                              directory=directory)
        except FileNotFoundError:
            print('One or more model files cannot be found. Terminating...')
            sys.exit()


class SaveCallback(Callback):
    def __init__(self, classifier):
        self.best_loss = 1000.0
        self.best_val_loss = 1000.0
        self.classifier = classifier
        super(SaveCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print()
        if logs['val_loss'] <= self.best_val_loss and logs['loss'] <= self.best_loss:
            self.classifier.save()
            self.best_val_loss = logs['val_loss']
            self.best_loss = logs['loss']

            self.classifier.log('Save point:\nLoss: %s\tAccuracy: %s\n' % (logs['loss'], logs['acc']) +
                                'Validation loss: %s\tValidation accuracy: %s\n' % (logs['val_loss'], logs['val_acc']),
                                out=False)
        elif logs['val_loss'] > self.best_val_loss:
            print('No improvement on validation loss. Skipping save...')
        elif logs['loss'] > self.best_loss:
            print('No improvement on loss. Skipping save...')


class TestCallback(Callback):
    def __init__(self, classifier, Xt, Xc, y):
        self.classifier = classifier
        self.Xt = Xt
        self.Xc = Xc
        self.y = y
        super(TestCallback, self).__init__()

    def on_train_end(self, logs=None):
        print('Evaluating on test set...')
        result = self.classifier.model.evaluate([self.Xt, self.Xc], self.y, batch_size=100)
        self.classifier.log('Test loss: %s --- Test acc: %s\n' % (result[0], result[1]))

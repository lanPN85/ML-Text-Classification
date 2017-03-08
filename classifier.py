from keras.layers.recurrent import GRU
from keras.layers.core import Dense
from keras.engine import merge
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input
from keras.callbacks import Callback

import utils
import numpy as np


class Classifier:
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
        content_inp = Input(shape=(content_len, ), name='Content_Input')
        content_embed = Embedding(input_dim=np.size(word_vec, 0), output_dim=np.size(word_vec, 1),
                                  weights=[word_vec], mask_zero=True, name='Content_Embedding')
        self.c_encoder = Sequential(name='Content_Encoder')
        self.c_encoder.add(content_embed)
        self.c_encoder.add(GRU(content_output, name='Content_GRU', consume_less='mem'))
        content_vec = self.c_encoder(content_inp)

        # Merge vectors to create output
        doc_vec = merge(inputs=[title_vec, content_vec], mode='concat')
        self.decoder = Sequential(name='Decoder')
        self.decoder.add(Dense(dense_neurons[0], input_shape=(title_output + content_output,),
                               name='Dense_0', activation='hard_sigmoid'))
        for i, n in enumerate(dense_neurons[1:]):
            self.decoder.add(Dense(n, activation='hard_sigmoid', name='Dense_%s' % (i+1)))
        self.decoder.add(Dense(len(classes), activation='softmax', name='Dense_Output'))
        output = self.decoder(doc_vec)

        self.model = Model(input=[title_inp, content_inp], output=output, name='Model')
        if weights is not None:
            self.model.load_weights(weights)

    def log(self, str, out=True):
        with open(self.directory + '/log.txt', 'at') as f:
            f.write(str)
        if out:
            print(str)

    def compile(self, optimizer=RMSprop, learning_rate=0.0001):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])

    def train(self, Xt, Xc, y, nb_epoch, batch_size=20):
        cb1 = SaveCallback(self)
        self.model.fit([Xt, Xc], y, nb_epoch=nb_epoch, batch_size=batch_size,
                       callbacks=[cb1], validation_split=0.2, shuffle=True)


class SaveCallback(Callback):
    def __init__(self, classifier):
        self.classifier = classifier
        super(SaveCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        utils.save_classifier(self.classifier, self.classifier.directory)


class TestCallback(Callback):
    def __init__(self, classifier, Xt, Xc, y):
        self.classifier = classifier
        self.Xt = Xt
        self.Xc = Xc
        self.y = y
        super(TestCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print('Evaluating on test set...')
        result = self.classifier.model.evaluate([self.Xt, self.Xc], self.y, batch_size=np.size(self.y, 0))
        self.classifier.log('Test loss: %s --- Test acc: %s' % (result[0], result[1]))

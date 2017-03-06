from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Merge
from keras.engine import merge
from keras.layers.embeddings import Embedding
from keras.optimizers import Adadelta
from keras.models import Model, Sequential
from keras.layers import Input

import numpy as np


class Classifier:
    def __init__(self, word_vec, word_to_index, index_to_word, classes, title_output=128, content_output=512,
                 dense_neurons=(1024, 256,), title_len=50, content_len=2000, weights=None):
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

    def compile(self, optimizer=Adadelta, learning_rate=0.0001):
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])


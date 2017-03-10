import utils
import sys
import datetime
import os

from keras.optimizers import *
from data_utils import *
from classifier import Classifier
from settings import *
from shutil import copy2

if len(sys.argv) > 1:
    DIRECTORY = './models/' + sys.argv[1]
else:
    DIRECTORY = "./models/%s" % datetime.date.today().isoformat()

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

if DATASET == 'ag_news':
    loader = load_ag_news
elif DATASET == 'bbc':
    loader = load_bbc
else:
    raise ValueError('Invalid dataset')

copy2('./settings.py', '%s/settings.py' % DIRECTORY)

matrices, word_vec, word_to_index, index_to_word, classes = loader(VOCABULARY_SIZE, TITLE_LEN, CONTENT_LEN)

print('Creating model...')
classifier = Classifier(word_vec, word_to_index, index_to_word, classes, title_output=TITLE_OUTPUT,
                        content_output=CONTENT_OUTPUT, dense_neurons=DENSE_NEURONS, title_len=TITLE_LEN,
                        content_len=CONTENT_LEN, directory=DIRECTORY, gru_regularize=GRU_LAMBDA,
                        dense_regularize=DENSE_LAMBDA)
classifier.compile(optimizer=RMSprop, learning_rate=LEARNING_RATE)

print('Starting training...')
classifier.train(matrices, N_EPOCH, batch_size=BATCH_SIZE)

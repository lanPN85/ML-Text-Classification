# Setup proper keras environment
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

import sys
import datetime

from timeit import default_timer as timer
from keras.optimizers import *
from data_utils import *
from classifier import Classifier
from settings import *
from shutil import copy2


# Get loader function for specified dataset
if DATASET == 'ag_news':
    loader = load_ag_news
elif DATASET == 'bbc':
    loader = load_bbc
elif DATASET == 'reuters':
    loader = load_reuters
else:
    raise ValueError('Invalid dataset')

# Set up the model's directory for saving and logging
# Allow override via command line argument
if len(sys.argv) > 1:
    DIRECTORY = './models/' + sys.argv[1]
else:
    DIRECTORY = "./models/%s_%s" % (DATASET, DENSE_NEURONS)
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
    os.makedirs(DIRECTORY + '/plots')
copy2('./settings.py', '%s/settings.py' % DIRECTORY)

print('Loading data...')
matrices, word_vec, word_to_index, index_to_word, classes = loader(VOCABULARY_SIZE, TITLE_LEN, CONTENT_LEN)

start_time = timer()

print('Creating model...')
classifier = Classifier(word_vec, word_to_index, index_to_word, classes, title_output=TITLE_OUTPUT,
                        content_output=CONTENT_OUTPUT, dense_neurons=DENSE_NEURONS, title_len=TITLE_LEN,
                        content_len=CONTENT_LEN, directory=DIRECTORY)
classifier.compile(optimizer=RMSprop, learning_rate=LEARNING_RATE)

print('Starting training...')
history = classifier.train(matrices, N_EPOCH, batch_size=BATCH_SIZE)

print('Plotting history...')
utils.plot_training(DIRECTORY + '/plots', history)

elapsed = timer() - start_time
print('Training complete in %s' % str(datetime.timedelta(seconds=elapsed)))

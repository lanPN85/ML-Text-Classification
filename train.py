import utils
import sys
import datetime
import os

from timeit import default_timer as timer
from keras.optimizers import *
from data_utils import *
from classifier import Classifier
from settings import *
from shutil import copy2

if DATASET == 'ag_news':
    loader = load_ag_news
elif DATASET == 'bbc':
    loader = load_bbc
elif DATASET == 'reuters':
    loader = load_reuters
else:
    raise ValueError('Invalid dataset')

if len(sys.argv) > 1:
    DIRECTORY = './models/' + sys.argv[1]
else:
    DIRECTORY = "./models/%s_%s" % (DATASET, datetime.date.today().isoformat())

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
    os.makedirs(DIRECTORY + '/plots')

copy2('./settings.py', '%s/settings.py' % DIRECTORY)

matrices, word_vec, word_to_index, index_to_word, classes = loader(VOCABULARY_SIZE, TITLE_LEN, CONTENT_LEN)
best_val, start_time = 0.0, timer()
for gru_lambda in GRU_LAMBDA:
    for dense_lambda in DENSE_LAMBDA:
        print('Training with regularization: %s %s' % (gru_lambda, dense_lambda))
        print('Creating model...')
        classifier = Classifier(word_vec, word_to_index, index_to_word, classes, title_output=TITLE_OUTPUT,
                                content_output=CONTENT_OUTPUT, dense_neurons=DENSE_NEURONS, title_len=TITLE_LEN,
                                content_len=CONTENT_LEN, directory=DIRECTORY, gru_regularize=gru_lambda,
                                dense_regularize=dense_lambda)
        classifier.compile(optimizer=RMSprop, learning_rate=LEARNING_RATE)

        print('Starting training...')
        prev_val, history = classifier.train(matrices, N_EPOCH, batch_size=BATCH_SIZE, prev_val_acc=best_val)
        if prev_val > best_val:
            best_val = prev_val
        print('Plotting history...')
        utils.plot_training(DIRECTORY + '/plots', history, gru_lambda, dense_lambda)

elapsed = timer() - start_time
print('Training complete in %s seconds' % str(datetime.timedelta(seconds=elapsed)))

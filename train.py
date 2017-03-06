import utils
import sys
import datetime
import os

from data_utils import *
from classifier import *
from settings import *

if len(sys.argv) > 0:
    DIRECTORY = sys.argv[0]
else:
    DIRECTORY = "./%s" % datetime.date.today().isoformat()

if os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

if DATASET == 'ag_news':
    loader = load_ag_news
elif DATASET == 'bbc':
    loader = load_bbc
else:
    raise ValueError('Invalid dataset')

matrices, word_vec, word_to_index, index_to_word, classes = loader(VOCABULARY_SIZE, TITLE_LEN, CONTENT_LEN)
classifier = Classifier(word_vec, word_to_index, index_to_word, classes, title_output=TITLE_OUTPUT,
                        content_output=CONTENT_OUTPUT, dense_neurons=DENSE_NEURONS, title_len=TITLE_LEN,
                        content_len=CONTENT_LEN)

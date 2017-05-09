# Setup proper keras environment
import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
keras.backend.set_image_dim_ordering('th')

import sys

from data_utils import *
from classifier import *

path, dataset = sys.argv[1], sys.argv[2]

if dataset == 'bbc':
    dataset = 'data/bbc_csv/test.csv'
elif dataset == 'ag_news':
    dataset = 'data/ag_news_csv/test.csv'
elif dataset == 'reuters':
    dataset = 'data/reuters_csv/test.csv'

classifier = utils.load_classifier(path, Classifier)
doc_list = load_csv(dataset, classifier.title_len, classifier.content_len)
Xt, Xc, y, unk, total = get_mat(doc_list, classifier.word_to_index, classifier.title_len,
                                classifier.content_len, len(classifier.classes))
print('Evaluating...')
acc, p, r, f1, loss = classifier.evaluate(Xt, Xc, y)
for i in range(len(p)-1):
    print('Precision @ label %s (%s): %s' % (i, classifier.classes[i], p[i]))
    print('Recall @ label %s (%s): %s' % (i, classifier.classes[i], r[i]))
    print('F1 Score: %s' % f1[i])
    print('------')

print('Mean Precision: %s' % p[-1])
print('Mean Recall: %s' % r[-1])
print('Mean F1: %s' % f1[-1])
print('Loss: %s' % loss)
print('Accuracy: %s' % acc)

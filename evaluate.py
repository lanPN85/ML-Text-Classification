import utils
import sys

from data_utils import *
from classifier import *

path, dataset = sys.argv[1], sys.argv[2]

if dataset == 'ag_news':
    loader = load_ag_news
elif dataset == 'bbc':
    loader = load_bbc
else:
    raise ValueError('Invalid dataset')

classifier = utils.load_classifier(path, Classifier)
matrices, word_vec, word_to_index, index_to_word, classes = loader(len(classifier.index_to_word)-2,
                                                                   classifier.title_len,
                                                                   classifier.content_len)
print('Evaluating...')
acc, p, r, f1 = classifier.evaluate(matrices['Xt_test'], matrices['Xc_test'], matrices['y_test'])
for i in range(len(p)-1):
    print('Precision @ label %s (%s): %s' % (i, classifier.classes[i], p[i]))
    print('Recall @ label %s (%s): %s' % (i, classifier.classes[i], r[i]))
    print('F1 Score: %s' % f1[i])
    print('------')

print('Mean Precision: %s' % p[-1])
print('Mean Recall: %s' % r[-1])
print('Mean F1: %s' % f1[-1])
print('Accuracy: %s' % acc)

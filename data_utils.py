import csv
import numpy as np
import utils
import nltk
import random
from glove import Glove
from keras.utils.np_utils import to_categorical


def load_embedding(vocabulary_size):
    print("Loading word embedding...")
    embed = Glove.load_stanford('data/glove.6B.100d.txt')
    embed_layer = np.asarray(embed.word_vectors[:vocabulary_size, :], dtype=np.float32)
    index_to_word = list(embed.inverse_dictionary.values())
    index_to_word.insert(0, utils.MASK_TOKEN)
    index_to_word = index_to_word[:vocabulary_size]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    word_count = len(index_to_word)
    index_to_word.append(utils.UNKNOWN_TOKEN)
    word_to_index[utils.UNKNOWN_TOKEN] = word_count + 1

    word_dim = np.size(embed_layer, 1)
    # Vector for the MASK token
    embed_layer = np.vstack((np.zeros((1, word_dim), dtype=np.float32), embed_layer))
    # Random vector for UNKNOWN_TOKEN, placed intentionally far away from vocabulary words
    embed_layer = np.vstack((embed_layer, np.asarray(np.random.uniform(20.0, 50.0, (1, word_dim)), dtype=np.float32)))

    return embed_layer, word_to_index, index_to_word


def _load_csv(filename, title_len, content_len):
    print('Reading %s...' % filename)
    f = open(filename, 'rt')
    reader = csv.reader(f, delimiter=',', quotechar='"')
    doc_list = []

    for row in reader:
        title = nltk.word_tokenize(row[1].lower())
        content = nltk.word_tokenize(row[2].lower())
        doc_list.append({'class': int(row[0]) - 1,
                         'title': title[:title_len],
                         'content': content[:content_len]})
    f.close()
    return doc_list


def strat_sample(doc_list, class_count, train_ratio=0.7):
    s = ([],) * class_count
    train_docs, test_docs = [], []
    for doc in doc_list:
        s[doc['class']].append(doc)

    for d in s:
        random.shuffle(d)
        train_split = int(len(d) * train_ratio)
        train_docs.extend(d[:train_split])
        test_docs.extend(d[train_split:])

    random.shuffle(train_docs)
    random.shuffle(test_docs)
    return train_docs, test_docs


def _get_mat(doc_list, word_to_index, word_vec, title_len, content_len, compress_labels=False):
    y = []
    Xt = np.zeros((len(doc_list), title_len), dtype=np.float32)
    Xc = np.zeros((len(doc_list), content_len), dtype=np.float32)

    total = 0.0
    unk = 0.0
    for i, doc in enumerate(doc_list):
        y.append(doc['class'])
        for j, word in enumerate(doc['title']):
            total += 1
            if word in word_to_index:
                idx = word_to_index[word]
            else:
                idx = word_to_index[utils.UNKNOWN_TOKEN]
                unk += 1
            Xt[i][j] = idx
        for j, word in enumerate(doc['content']):
            total += 1
            if word in word_to_index:
                idx = word_to_index[word]
            else:
                idx = word_to_index[utils.UNKNOWN_TOKEN]
                unk += 1
            Xc[i][j] = idx

    y = np.asarray(y, dtype=np.int32)
    if not compress_labels:
        y = to_categorical(y, np.max(y)+1)
    return Xt, Xc, y, unk, total


def load_ag_news(vocabulary_size, title_len, content_len, path='./data/ag_news_csv'):
    with open(path + '/classes.txt', 'rt') as f:
        classes = f.readlines()
        for i in range(len(classes)):
            classes[i] = classes[i].rstrip()
        f.close()

    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)
    train_docs = _load_csv(path + '/train.csv', title_len, content_len)
    Xt_train, Xc_train, y_train, unk1, total1 = _get_mat(train_docs, word_to_index, embed_layer, title_len, content_len)
    test_docs = _load_csv(path + '/test.csv', title_len, content_len)
    Xt_test, Xc_test, y_test, unk2, total2 = _get_mat(test_docs, word_to_index, embed_layer, title_len, content_len)

    print('%d unknown tokens / %d tokens' % (unk1 + unk2, total1 + total2))
    print('Unknown token ratio: %s%%' % ((unk1 + unk2)*100 / (total1 + total2)))

    matrices = {'Xt_train': Xt_train,
                'Xc_train': Xc_train,
                'y_train': y_train,
                'Xt_test': Xt_test,
                'Xc_test': Xc_test,
                'y_test': y_test}
    return matrices, embed_layer, word_to_index, index_to_word, classes


def load_bbc(vocabulary_size, title_len, content_len, path='./data/bbc_csv'):
    with open(path + '/classes.txt', 'rt') as f:
        classes = f.readlines()
        for i in range(len(classes)):
            classes[i] = classes[i].rstrip()
        f.close()

    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)
    docs = _load_csv(path + '/train.csv', title_len, content_len)
    print('Parsed %s documents.' % len(docs))
    train_docs, test_docs = strat_sample(docs, len(classes), train_ratio=0.7)
    Xt_train, Xc_train, y_train, unk1, total1 = _get_mat(train_docs, word_to_index, embed_layer, title_len, content_len)
    Xt_test, Xc_test, y_test, unk3, total3 = _get_mat(test_docs, word_to_index, embed_layer, title_len, content_len)

    unk, total = unk1 + unk3, total1 + total3
    print('%d unknown tokens / %d tokens' % (unk, total))
    print('Unknown token ratio: %s%%' % (unk * 100 / total))

    matrices = {'Xt_train': Xt_train,
                'Xc_train': Xc_train,
                'y_train': y_train,
                'Xt_test': Xt_test,
                'Xc_test': Xc_test,
                'y_test': y_test,
                }
    return matrices, embed_layer, word_to_index, index_to_word, classes

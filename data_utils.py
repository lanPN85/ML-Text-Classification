import csv
import numpy as np
import utils
import nltk
from glove import Glove


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
        title = nltk.word_tokenize(row[1])
        content = nltk.word_tokenize(row[2])
        doc_list.append({'class': int(row[0]) - 1,
                         'title': title[:title_len],
                         'content': content[:content_len]})
    f.close()
    return doc_list


def _get_mat(doc_list, word_to_index, word_vec, title_len, content_len):
    y = []
    Xt = np.zeros((len(doc_list), title_len), dtype=np.float32)
    Xc = np.zeros((len(doc_list), content_len), dtype=np.float32)
    for i, doc in enumerate(doc_list):
        y.append(doc['class'])
        for j, word in enumerate(doc['title']):
            if word in word_to_index:
                idx = word_to_index[word]
            else:
                idx = word_to_index[utils.UNKNOWN_TOKEN]
            Xt[i][j] = idx
        for j, word in enumerate(doc['content']):
            if word in word_to_index:
                idx = word_to_index[word]
            else:
                idx = word_to_index[utils.UNKNOWN_TOKEN]
            Xc[i][j] = idx

    y = np.asarray(y, dtype=np.int32)
    return Xt, Xc, y


def load_ag_news(vocabulary_size, title_len, content_len, path='./data/ag_news_csv'):
    with open(path + '/classes.txt', 'rt') as f:
        classes = f.readlines()
        f.close()

    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)
    train_docs = _load_csv(path + '/train.csv', title_len, content_len)
    Xt_train, Xc_train, y_train = _get_mat(train_docs, word_to_index, embed_layer, title_len, content_len)
    test_docs = _load_csv(path + '/test.csv', title_len, content_len)
    Xt_test, Xc_test, y_test = _get_mat(test_docs, word_to_index, embed_layer, title_len, content_len)
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
        f.close()

    embed_layer, word_to_index, index_to_word = load_embedding(vocabulary_size)
    train_docs = _load_csv(path + '/train.csv', title_len, content_len)
    Xt_train, Xc_train, y_train = _get_mat(train_docs, word_to_index, embed_layer, title_len, content_len)
    matrices = {'Xt_train': Xt_train,
                'Xc_train': Xc_train,
                'y_train': y_train}
    return matrices, embed_layer, word_to_index, index_to_word, classes

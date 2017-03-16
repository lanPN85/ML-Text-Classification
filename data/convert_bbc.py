import os
import random
import numpy as np

DIRECTORY = './bbc_csv'
SOURCE_DIR = './bbc'
if not os.path.exists(DIRECTORY):
    print('Creating %s' % DIRECTORY)
    os.mkdir(DIRECTORY)

topic_dirs = os.listdir(SOURCE_DIR)
topic_dirs.remove('README.TXT')
docs = [[]] * len(topic_dirs)
train_docs, test_docs = [], []
f_class = open(DIRECTORY + '/classes.txt', mode='wt')
f_train = open(DIRECTORY + '/train.csv', mode='wt')
f_test = open(DIRECTORY + '/test.csv', mode='wt')

for idx, directory in enumerate(topic_dirs):
    f_class.write(directory.capitalize() + '\n')
    files = os.listdir(SOURCE_DIR + '/' + directory)
    cls = idx + 1

    for fn in files:
        f = open(SOURCE_DIR + '/' + directory + '/' + fn, 'rt')
        try:
            lines = f.readlines()
        except UnicodeDecodeError:
            f.close()
            continue
        f.close()
        title = lines[0].strip()
        content = ''
        for line in lines[1:]:
            content += line.strip() + ' '
        tmp = docs[idx].copy()
        tmp.append({
            'class': cls,
            'content': content,
            'title': title
        })
        docs[idx] = tmp

for d in docs:
    random.shuffle(d)
    train_split = int(len(d) * 0.8)
    train_docs.extend(d[:train_split])
    test_docs.extend(d[train_split:])
random.shuffle(train_docs)
random.shuffle(test_docs)

print('Creating training set...')
for d in train_docs:
    out = '"%s","%s","%s"\n' % (d['class'], d['title'], d['content'])
    print(out)
    f_train.write(out)

print('Creating test set...')
for d in test_docs:
    out = '"%s","%s","%s"\n' % (d['class'], d['title'], d['content'])
    print(out)
    f_test.write(out)

f_class.close()
f_train.close()
f_test.close()

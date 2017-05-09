import os
import csv
import random
import shutil


def load_csv(filename):
    print('Reading %s...' % filename)
    f = open(filename, 'rt')
    reader = csv.reader(f, delimiter=',', quotechar='"')
    doc_list = []

    for row in reader:
        title = row[1]
        content = row[2]
        doc_list.append({'class': int(row[0]),
                         'title': title,
                         'content': content})
        print(' '.join(row))
    f.close()
    return doc_list


def strat_sample(doc_list, class_count, train_ratio=0.8):
    s = [[]] * class_count
    train_docs, val_docs = [], []
    for d in doc_list:
        tmp = s[d['class']-1].copy()
        tmp.append(d)
        s[d['class']-1] = tmp

    for d in s:
        random.shuffle(d)
        train_split = int(len(d) * train_ratio)
        train_docs.extend(d[:train_split])
        val_docs.extend(d[train_split:])

    random.shuffle(train_docs)
    random.shuffle(val_docs)
    return train_docs, val_docs


docs = []
docs.extend(load_csv('./ag_news/test.csv'))
docs.extend(load_csv('./ag_news/train.csv'))

train, test = strat_sample(docs, 4, 0.7)
os.makedirs('./ag_news_csv', exist_ok=True)

ftrain = open('./ag_news_csv/train.csv', mode='wt', newline='')
ftest = open('./ag_news_csv/test.csv', mode='wt', newline='')
writer1 = csv.writer(ftrain, delimiter=',', quotechar='"')
writer2 = csv.writer(ftest, delimiter=',', quotechar='"')

for doc in train:
    writer1.writerow([doc['class'], doc['title'], doc['content']])

for doc in test:
    writer2.writerow([doc['class'], doc['title'], doc['content']])

shutil.copy2('./ag_news/classes.txt', './ag_news_csv/classes.txt')
ftrain.close()
ftest.close()

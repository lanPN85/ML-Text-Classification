import feedparser

from demo_rss.article import Article


def parse(url):
    doc = feedparser.parse(url)
    source = doc.feed.title
    articles = []

    for entry in doc.entries:
        a = Article()
        a.title = entry.title
        try:
            a.content = entry.description
        except:
            pass
        a.source = source
        a.url = entry.link
        articles.append(a)

    return articles


def classify(classifier, articles):
    for a in articles:
        pred, probs = classifier.predict(a.title, a.content)
        a.topic = classifier.classes[pred]
        a.topic_dist = dict([(c, probs[0][i]) for i, c in enumerate(classifier.classes)])
        print('\t%s\t%s' % (a.url, a.topic))

    return articles


def store(articles, filepath):
    with open(filepath, 'at') as f:
        f.write('"URL","Source","Title","Content","Topic","Distribution"\n')
        for a in articles:
            f.write('"%s","%s","%s","%s","%s","%s"\n' %
                    (a.url, a.source, a.title, a.content, a.topic,
                     a.topic_dist))
        f.close()

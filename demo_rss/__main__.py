import sys
import utils

from classifier import Classifier
from demo_rss import pipeline

feed_list = [
    'http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'http://rss.cnn.com/rss/edition.rss',
    'https://www.theguardian.com/uk/rss',
]
filepath = './demo_rss/results.csv'
model = sys.argv[1]
classifier = utils.load_classifier(model, Classifier)

for feed in feed_list:
    print('Parsing %s...' % feed)
    pipeline.store(pipeline.classify(classifier, pipeline.parse(feed)), filepath)

print('Complete. View results at %s' % filepath)

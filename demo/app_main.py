from kivy.app import App
from kivy.config import Config
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from classifier import Classifier

import utils
import matplotlib
import heapq
import os
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Root(BoxLayout):
    title_inp = ObjectProperty()
    content_inp = ObjectProperty()
    model_desc = ObjectProperty()
    pred_res = ObjectProperty()
    graph = ObjectProperty()
    chooser = ObjectProperty()
    config = ObjectProperty()

    def __init__(self):
        super(Root, self).__init__()


class NewsClassifier(App):
    APP_NAME = 'News Classifier'
    classifier = None

    @staticmethod
    def init_graph():
        plt.close()
        plt.title('Prediction breakdown')
        plt.xlim(0.0, 1.0)
        plt.xticks(np.linspace(0, 1.1, 12))
        plt.xlabel('Confidence')

        plt.ylabel('Class')

    def empty_graph(self):
        self.root.graph.nocache = True
        self.init_graph()
        short_list = [''] + self.classifier.classes[:7] + ['']
        plt.ylim(0, len(short_list) - 1)
        plt.yticks(range(len(short_list)), short_list)
        plt.savefig('./current_empty.png')

    def plot_result(self, probs):
        top7 = heapq.nlargest(7, range(len(probs[0])), probs[0].take)[::-1]
        top_labels = [''] + [self.classifier.classes[i] for i in top7] + ['']
        labels_to_val = {}
        for i in top7:
            labels_to_val[self.classifier.classes[i]] = probs[0][i]

        plt.yticks(range(len(top_labels)), top_labels)
        for i in range(1, len(top_labels)-1):
            plt.plot([0, labels_to_val[top_labels[i]]], [i, i], linewidth=20)

        plt.savefig('./current.png')

    def get_prediction(self):
        if self.classifier is None:
            return
        title = self.root.title_inp.text
        content = self.root.content_inp.text
        pred, probs = self.classifier.predict(title, content)

        self.root.graph.source = './current_empty.png'
        self.plot_result(probs)
        self.root.pred_res.text = self.classifier.classes[pred]
        self.empty_graph()
        self.root.graph.source = './current.png'

    def clear_text(self):
        if self.classifier is not None:
            self.empty_graph()
            if os.path.exists('./current.png'):
                os.remove('./current.png')
            self.root.pred_res.text = ''

            self.root.graph.source = './current_empty.png'

    def load_classifier(self):
        path = self.root.chooser.selection[0]
        self.classifier = utils.load_classifier(path, Classifier)
        self.empty_graph()
        self.root.graph.source = './current_empty.png'

        with open(path + '/settings.py', 'rt') as f:
            conf = f.readlines()
            self.root.config.text = ''.join(conf)

        name = path.split('/')[-1]
        self.root.model_desc.text = name


def run():
    Config.set('graphics', 'width', '1600')
    Config.set('graphics', 'height', '900')
    Config.set('graphics', 'minimum_width', '1600')
    Config.set('graphics', 'minimum_height', '900')
    NewsClassifier().run()


if __name__ == '__main__':
    run()

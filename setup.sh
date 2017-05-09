#!/usr/bin/env bash
sudo apt-get install python3-pip
sudo pip3 install keras==1.2.2 h5py glove-python nltk matplotlib
sudo python3 -m nltk.downloader punkt
cp -r ./.keras ~

#wget http://nlp.stanford.edu/data/glove.6B.zip ./data/
#unzip -j ./data/glove.6B.zip glove.6B.100d.txt -d ./data/

sudo pip3 install kivy feedparser

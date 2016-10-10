#!/bin/sh

sudo apt-get update


sudo apt-get install python-dev python-pip csh perl libgsl0-dev python-scipy r-base

sudo pip install nltk
sudo pip install gensim

sudo python -m nltk.downloader -d /usr/share/nltk_data punkt
sudo python -m nltk.downloader -d /usr/share/nltk_data averaged_perceptron_tagger
sudo python -m nltk.downloader -d /usr/share/nltk_data universal_tagset

sudo cpan > install Algorithm::Munkres

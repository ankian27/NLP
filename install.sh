#!/bin/sh

sudo apt-get update


sudo apt-get install python-dev python-pip csh

sudo pip install nltk

# install nltk data packages stopwords and punkt used for word tokenizer

sudo python -m nltk.downloader -d /usr/share/nltk_data stopwords

sudo python -m nltk.downloader -d /usr/share/nltk_data punkt

sudo cpan > install Algorithm::Munkres

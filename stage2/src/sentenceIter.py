import nltk
import gensim
import re
import string
import os

class SentenceIter():
    def __init__(self, d_name):
        self.dir_name = d_name

    def __iter__(self):
        file_num = 0
        for file_name in os.listdir(self.dir_name):
            print 'Processing ' + file_name + ', number: ' + str(file_num)
            file_num += 1
            for sentence in self.get_sentences(self.dir_name + '/' + file_name):
                yield sentence

    """
    file_name should have one sentence per line
    """
    def get_sentences(self, file_name):
        with open(file_name, 'r') as f_ref:
            for line in f_ref:
                line = line.strip()
                # nltk can't handle non-ascii characters, so strip them
                # out
                line = filter(lambda c: 0 < ord(c) < 127, line)
                yield filter(lambda token: any(c.isalpha() for c in token),
                        nltk.word_tokenize(line))

if __name__ == '__main__':
    si = SentenceIter('../../../sentences')
    for s in si:
        print s

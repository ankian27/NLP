from gensim import corpora
from gensim.similarities.docsim import Similarity
from nltk.stem.porter import PorterStemmer
import xmltodict
import string
import nltk
import re

stemmer = PorterStemmer()
sws = set(line.strip() for line in open('stopwords.txt', 'r'))
non_alnum = re.compile('[\W_]+')
split_re = re.compile(r'<head>.*</head>')

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def tokenize(context, ctx_len):
    context = strip_non_ascii(context)
    b4, after = split_re.split(context)
    doc = []
    tmp1 = []
    tmp2 = []
    for word in nltk.word_tokenize(b4):
        if '-' in word:
            for word1 in word.split('-'):
                if word1.lower() not in sws:
                    tmp1.append(non_alnum.sub('', stemmer.stem(word1)))
        elif any(char in word for char in string.punctuation + string.digits):
            continue
        else:
            if word.lower() not in sws:
                tmp1.append(non_alnum.sub('', stemmer.stem(word)))
    doc = doc[-ctx_len:]
    for word in nltk.word_tokenize(after):
        if '-' in word:
            for word1 in word.split('-'):
                if word1.lower() not in sws:
                    tmp2.append(non_alnum.sub('', stemmer.stem(word1)))
        elif any(char in word for char in string.punctuation + string.digits):
            continue
        else:
            if word.lower() not in sws:
                tmp2.append(non_alnum.sub('', stemmer.stem(word)))

    for word in reversed(tmp1):
        if word:
            doc.append(word.lower())
        if len(doc) >= ctx_len:
            break
    for word in tmp2:
        if word:
            doc.append(word.lower())
        if len(doc) >= ctx_len*2:
            break
        
    return doc

class Corp():

    """
    @param f: an xml file in sense2eval format
    """
    def __init__(self, corp):
        return

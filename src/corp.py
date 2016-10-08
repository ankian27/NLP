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
split_re = re.compile(r'<head>(.*?)</head>')
rm_head_re = re.compile(r'</?head>')

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def tokenize(context, ctx_len):
    context = strip_non_ascii(context)
    b4, target, after = split_re.split(context, 1)
    after = rm_head_re.sub('', after)
    #print "before: " + b4
    #print "after: " + after
    #print "target: " + target
    doc = []
    tmp1 = []
    tmp2 = []
    for word in nltk.word_tokenize(b4):
        # nltk doesn't split hyphenated words for us
        if '-' in word:
            for word1 in word.split('-'):
                    tmp1.append(word1)
        else:
            tmp1.append(word)
    for word in nltk.word_tokenize(after):
        # nltk doesn't split hyphenated words for us
        if '-' in word:
            for word1 in word.split('-'):
                    tmp2.append(word1)
        else:
            tmp2.append(word)

    for word in reversed(tmp1):
        word = word.lower()
        if word not in sws:
            doc.append(word)
        if len(doc) >= ctx_len:
            break
    doc.reverse()
    doc.append(target.lower())
    for word in tmp2:
        word = word.lower()
        if word not in sws:
            doc.append(word)
        if len(doc) >= ctx_len*2:
            break
        
    return doc

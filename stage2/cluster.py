import sys
import math
import re
import nltk
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
import numpy as np

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


model_file = 'models/3_parts_of_wiki_lowercase'
MIN_VOC_FREQ = 5

def getCtxes( f):
    #RE1
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

def get_tokens(file_name, target_word, model, stopwords):
    rename_re = re.compile(r'<head>(.*?)</head>')
    rm_head_re = re.compile(r'</?head>')
    ctx_tokens = []
    senses = []
    for sense, ctx in getCtxes(file_name):
        ctx = rename_re.sub(r' \1 ', ctx, 1)
        ctx = rm_head_re.sub('', ctx)
        ctx = filter(lambda c: 0 < ord(c) < 127, ctx)
        tokens = set()
        for token in nltk.word_tokenize(ctx):
            if not token: 
                continue 
            elif not any(c.isalpha() for c in token):
                continue
            elif '\'' in token:
                continue
            elif '-' in token:
                # nltk doesn't split hyphenated words for us
                tokens |= set(t.lower() for t in token.split('-'))
            else:
                tokens.add(token.lower())
        ctx_tokens.append(filter(lambda x: x in model and x not in stopwords and target_word not in x, tokens))
        senses.append(sense)
    return ctx_tokens, senses

def cos_sim2dist(similarity):
    if similarity > 1:
        sim = 1
    elif similarity < -1:
        sim = -1
    else:
        sim = similarity
    return (math.acos(sim)/math.pi)

def calc_distances(ctxes, model):
    distances = []
    for ctx1 in ctx_tokens:
        distances.append([])
        for ctx2 in ctx_tokens:
            distances[-1].append(cos_sim2dist(model.n_similarity(ctx1, ctx2)))
    return distances

'''
returns a set of words which encompasses the vocab extracted from the ctxes.
'''
def build_vocab(ctxes):
    # word-by-document matrix
    matrix = []
    word_counts = defaultdict(int)
    for ctx_tokens in ctxes:
        for token in ctx_tokens:
            word_counts[token] += 1
    #words = [(word, count) for word, count in word_counts.iteritems()]
    #words = filter(lambda x: x[1] > 4, words)
    #words = sorted(words, key=lambda x: x[1])
    #print '\n'.join(word + ': ' + str(count) for word, count in words)
    return set(word for word, count in word_counts.iteritems() if count >= MIN_VOC_FREQ)

def reduce_ctxes(ctxes, vocab):
    reduced_ctxes = []
    for ctx_tokens in ctxes:
        reduced_ctxes.append(filter(lambda token: token in vocab, ctx_tokens))
    return reduced_ctxes

def main(file_name):
    target_word, pos = file_name.split('-', 1)
    target_word = target_word.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        target = "xyz"
        pos = "noun"
    
    model = Word2Vec.load(model_file)
    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))
    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    ctxes, senses = get_tokens(file_name, target_word, model, stopwords)

    vocab = build_vocab(ctxes)

    reduced_ctxes = reduce_ctxes(ctxes, vocab)
    for i, r_ctx in enumerate(reduced_ctxes):
        print 'ctx id: ' + str(i) + ' ' + ' '.join(r_ctx)

#    clusters = defaultdict(list)
#    for i, label in enumerate(db.labels_):
#        clusters[label].append(senses[i])
#    for label, ctxes in clusters.iteritems():
#        print str(label)
#        print ' '.join(ctxes)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python writeToHDP.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

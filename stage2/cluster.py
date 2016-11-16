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
MIN_VOC_FREQ = 1
TAR_WINDOW = 2
WORD_VEC_SIZE = 5

def get_ctxes( f):
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
    for sense, ctx in get_ctxes(file_name):
        ctx = rename_re.sub(' ' + target_word + 'target ', ctx, 1)
        ctx = rm_head_re.sub('', ctx)
        ctx = filter(lambda c: 0 < ord(c) < 127, ctx)
        tokens = []
        for token in nltk.word_tokenize(ctx):
            if not token: 
                continue 
            elif not any(c.isalpha() for c in token):
                continue
            elif '\'' in token or '.' in token:
                continue
            elif '-' in token:
                # nltk doesn't split hyphenated words for us
                tokens.extend([t.lower() for t in token.split('-')])
            else:
                tokens.append(token.lower())
        # index function will throw an error if we don't find the target
        # word
        target_word_i = tokens.index(target_word + 'target')
        ctx_tokens.append(set(filter(lambda x: x in model and x not in stopwords and target_word not in x, tokens)))
        # add the nearest 2 words to the target word to the ctx
        ctx_tokens[-1].update(tokens[target_word_i - TAR_WINDOW:target_word_i])
        ctx_tokens[-1].update(tokens[target_word_i + 1:target_word_i + TAR_WINDOW + 1])
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
    for ctx1 in ctxes:
        distances.append([])
        for ctx2 in ctxes:
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
#    words = [(word, count) for word, count in word_counts.iteritems()]
#    words = filter(lambda x: x[1] >= MIN_VOC_FREQ, words)
#    words = sorted(words, key=lambda x: x[1])
#    print '\n'.join(word + ': ' + str(count) for word, count in words)
    vocab = set(word for word, count in word_counts.iteritems() if count >= MIN_VOC_FREQ)
    word_counts = {word: count for word, count in word_counts.iteritems() if word in vocab}
    return vocab, word_counts

'''
ctxes is a list of sets of words.
'''
def reduce_ctxes(ctxes, vocab, counts):
    reduced_ctxes = []
    for ctx_tokens in ctxes:
        context_words = []
        for token in ctx_tokens:
            if token in vocab:
                context_words.append((token, counts[token]))
        context_words = sorted(context_words, key=lambda x: x[1], reverse=True)
        reduced_ctxes.append(set(word for word, _ in context_words[0:WORD_VEC_SIZE]))
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

    # read the stopwords into a set
    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    ctxes, senses = get_tokens(file_name, target_word, model, stopwords)

    vocab, counts = build_vocab(ctxes)

    reduced_ctxes = reduce_ctxes(ctxes, vocab, counts)
    for i, r_ctx in enumerate(reduced_ctxes):
        print 'ctx id: ' + str(i) + ' ' + ' '.join(r_ctx)
    return
    distances = calc_distances(reduced_ctxes, model)
    #for l in distances:
        #print l

    db = DBSCAN(eps=0.3, min_samples=10, metric='precomputed', n_jobs=-1).fit(distances) 
    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        clusters[label].append(senses[i])
    for label, ctxes in clusters.iteritems():
        print str(label)
        print ' '.join(ctxes)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python writeToHDP.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

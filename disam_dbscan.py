from src.corp import Corp
from src.corp import tokenize
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities.docsim import Similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict
import xmltodict
import optparse
import sys
import re

def getCtxes(f):
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

"""
Disambiguates the given sense2eval xml file, f. The target parameter is only necessary if
the target word isn't marked with <head> tags.
@param f: the file path to the sense2eval xml formatted file
@param target: optional target word
"""
def disam(f, target=None):
    docs = []
    counts = defaultdict(int)
    for sense, ctx in getCtxes(f):
        for word in tokenize(ctx, 10):
            counts[word] += 1
        docs.append((sense, tokenize(ctx, 500)))
    top = []
    for word, count in counts.iteritems():
        top.append((word, count))
    #print sorted(top, key=lambda tup: tup[1], reverse=True)
    features = set()
    for word, count in sorted(top, key=lambda tup: tup[1], reverse=True)[:50]:
        features.add(word)

    # work around to make dictionary
    tmp = []
    tmp.append(list(features))
    dictionary = corpora.Dictionary(tmp)

    corp = []
    for i, tup in enumerate(docs):
        sense, doc = tup
        doc = set(word for word in doc if word in features)
        docs[i] = (sense, list(doc))
        #corp.append(doc)
        corp.append(dictionary.doc2bow(doc))
    index = Similarity('./', corp, num_features=len(dictionary))
    distances = []
    for sims in index:
        distances.append([1 - sim for sim in sims])
    #print distances
    db = DBSCAN(eps=0.5, min_samples=10, metric="precomputed").fit(distances)
    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        clusters[label].append(docs[i])
    for label, ctxes in clusters.iteritems():
        print str(label)
        print '\n'.join(sense + ": " + ' '.join(ctx) for sense, ctx in ctxes)

if __name__ == '__main__':
    parser = optparse.OptionParser(description='Disambiguate a target word.')
    parser.add_option('-t', dest='target', action='store',
                        default=None,
                        help='the target word')
    # first arg is the program name. Ignore it
    (options, f) = parser.parse_args(sys.argv[1:])
    if not f:
        print "need an xml file to parse"
        sys.exit(1)
    disam(f[0], options.target)


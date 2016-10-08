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

def writeToHDP(f, corp):
    target, pos = f.split('-', 1)
    target = target.split('/')[-1]
    with open("hdp-wsi-master/wsi_input/example/all/" + target + "." + pos[0] + ".lemma", 'w+') as f_ref:
        for doc in corp:
            f_ref.write(' '.join(word for word in doc) + "\n")
            #hdp format
            #print str(len(doc)) + " " + ' '.join(str(term) + ":" + str(count) for term, count in doc)

"""
Disambiguates the given sense2eval xml file, f. The target parameter is only necessary if
the target word isn't marked with <head> tags.
@param f: the file path to the sense2eval xml formatted file
@param target: optional target word
"""
def disam(f, target=None):
    docs = []
    counts = defaultdict(int)

    # Process each context
    for sense, ctx in getCtxes(f):
        # Get word counts for words that occur within X words of target
        # word
        for word in tokenize(ctx, 3):
            counts[word] += 1
        # Save the entire tokenized context
        docs.append((sense, tokenize(ctx, 500)))

    writeToHDP(f, [doc for sense, doc in docs])
    return
    top = []
    # Put word counts into a list so we can sort them
    for word, count in counts.iteritems():
        top.append((word, count))
    features = set()

    # Get our feature words based on words with highest counts
    for word, count in sorted(top, key=lambda tup: tup[1], reverse=True)[:50]:
        features.add(word)

    # Remove any words from each context that are not feature words 
    corp = []
    for i, tup in enumerate(docs):
        sense, doc = tup
        doc = set(word for word in doc if word in features)
        docs[i] = (sense, list(doc))
        corp.append(list(doc))
        
    # Write out hdp-wsi format
    return
    for doc in corp:
        terms = defaultdict(int)
        for word in doc:
            terms[word] += 1
        print str(len(terms)) + " " + ' '.join(str(term) + ":" + str(count) for term, count in terms.iteritems())
    return
    index = Similarity('./', corp, num_features=len(dictionary))
    distances = []
    for sims in index:
        distances.append([1 - sim for sim in sims])
    print distances
    db = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(distances)
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


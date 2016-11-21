import sys
import re
import nltk
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation

model_file = 'models/3_parts_of_wiki_lowercase'
WINDOW_SIZE = 10
MIN_VOC_FREQ = 2
WORD_VEC_SIZE = 5

def get_ctxes(f):
    #RE1
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

"""
Convert the contexts in a sense2val file into a tokenized list. Also returns the senses associated with each context.
"""
def tokenize_ctxes(file_name, target_word, stopwords, window_size):
    rename_re = re.compile(r'<head>(.*?)</head>')
    rm_head_re = re.compile(r'</?head>')
    ctx_tokens = []
    senses = []
    for sense, ctx in get_ctxes(file_name):
        ctx = rename_re.sub(' ' + target_word + 'target ', ctx, 1)
        ctx = rm_head_re.sub('', ctx)
        ctx = filter(lambda c: 0 < ord(c) < 127, ctx)
        tokens = []
        # Convert the context into an ordered list of tokens. The following
        # types of tokens are removed in this step:
        # - stopwords
        # - tokens consisting of only punctuation
        # - tokens with no alphabetical characters
        # - tokens with a '.' in them (this indicates an acronym which
        #   is generally not useful)
        # - tokens with single quote in them (these are usually the
        #   second half of a conjuction, for example "n't" or "'ve"
        #   which are not usually useful)
        for token, tag in nltk.pos_tag(nltk.word_tokenize(ctx)):
            token = token.lower()
            if not token or token in stopwords: 
                continue 
            elif not any(c.isalpha() for c in token):
                continue
            elif '\'' in token or '.' in token:
                continue
            elif '-' in token:
                # nltk doesn't split hyphenated words for us
                for t in token.split('-'):
                    if t:
                    	tokens.append((t, nltk.map_tag('en-ptb', 'universal', tag)))
            else:
                tokens.append((token, nltk.map_tag('en-ptb', 'universal', tag)))
        # index function will throw an error if we don't find the target
        # word
        target_word_i = 0
        while tokens[target_word_i][0] != target_word + 'target':
            target_word_i += 1
        ctx_tokens.append(tokens[target_word_i - window_size:target_word_i] + tokens[target_word_i + 1:target_word_i + window_size + 1])
        senses.append(sense)
    return ctx_tokens, senses

def make_context_vecs(ctxes, model):
    vecs = []
    for ctx in ctxes:
        vec = None
        #weight = float(float(1)/float(len(ctx)))
        weight = float(1)
        for word in ctx:
            if word not in model:
                continue
            if vec is None:
                vec = model[word]*weight
            else:
                vec += model[word]*weight
        vecs.append(vec)
    return vecs

def get_clusters(cluster, senses, ctxes):
    clusters = []
    # cluster.label -> index in clusters
    clusterlabel_to_index = {}
    for i, label in enumerate(cluster.labels_):
        if label not in clusterlabel_to_index:
            clusterlabel_to_index[label] = len(clusters)
            clusters.append([])
        clusters[clusterlabel_to_index[label]].append((senses[i], ctxes[i]))
    return clusters

def print_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print 'Cluster ' + str(i)
        print '----------'
        print '\n'.join(' '.join(context) for _, context in cluster)

def main(file_name):
    target_word, pos = file_name.split('-', 1)
    target_word = target_word.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        target = "xyz"
        pos = "noun"

    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    model = Word2Vec.load(model_file)

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE)
    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append([word for word, _ in ctx])
    ctx_vecs = make_context_vecs(raw_ctxes, model)

    #for i, context in enumerate(ctxes):
        #print 'Context ' + str(i) + '| sense: ' + senses[i] + ' | words: ' + ' '.join(str(item) for item in context)
    #for vec in ctx_vecs:
        #print vec

    ap = AffinityPropagation(damping=0.5, convergence_iter=15, max_iter=300).fit(ctx_vecs)

    clusters = get_clusters(ap, senses, raw_ctxes)

    print_clusters(clusters)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python def_gen.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

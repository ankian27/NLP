import sys, re, nltk, os, math, traceback
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import numpy as np
from src.DefinitionGeneration import Definition

model_file = 'models/Google'
WINDOW_SIZE = 10
MIN_VOC_FREQ = 2
TOP_N = 50
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
        # will break if target word is 'target'
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
            elif target_word in token and token != target_word + 'target':
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
        if len(ctx) == 0:
            vecs.append(model['the'] - model['the'])
            continue
        weight = float(float(1)/float(len(ctx)))
        #weight = float(1)
        vec = None
        for word in ctx:
            if vec is None:
                vec = model[word]*weight
            else:
                vec += model[word]*weight
        vecs.append(vec)
    return vecs

def word_stats_lists(ctxes):
    total_count = 0
    word_counts = defaultdict(int)
    for ctx in ctxes:
        tot_ctx_words = 0
        total_count += len(ctx)
        for word in ctx:
            word_counts[word] += 1
    return total_count, word_counts

def calc_distances(ctxes, model):
    distances = []
    for ctx1 in ctxes:
        distances.append([])
        for ctx2 in ctxes:
            if len(ctx1) == 0 or len(ctx2) == 0:
                distances[-1].append(float(1))
            elif isinstance(ctx2, str):
                distances[-1].append(1 - model.similarity(ctx1, ctx2))
            else:
                distances[-1].append(1 - model.n_similarity(ctx1, ctx2))
    return distances

def make_key(f, target_word, pos):
    thing = target_word + '.' + pos[0].lower()
    ctx_re = re.compile(r'<instance id="[0-9]+">.*?<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>.*?</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    mapping = {}
    cur_id = 1
    with open('senseclusters_scorer/key', 'w+') as key_ref:
        for i, sense in enumerate(ctx_re.findall(buf)):
            if sense not in mapping:
                mapping[sense] = cur_id
                cur_id += 1
            key_ref.write(thing + " " + thing + "." + str(i) + " " + thing + "." + str(mapping[sense]) + "\n")

def make_answers(cluster_obj, target_word, pos):
    thing = target_word + '.' + pos[0].lower()
    with open('senseclusters_scorer/answers', 'w+') as ans_ref:
        for i, label in enumerate(cluster_obj.labels_):
            ans_ref.write(thing + " " + thing + "." + str(i) + " " + thing + "." + str(label + 1) + "\n")

def get_clusters(cluster, words):
    # each item is a list of contexts representing a cluster
    clusters = []
    # cluster.label -> index in clusters
    clusterlabel_to_index = {}
    for i, label in enumerate(cluster.labels_):
        if label not in clusterlabel_to_index:
            clusterlabel_to_index[label] = len(clusters)
            clusters.append([])
        clusters[clusterlabel_to_index[label]].append(words[i])
    return clusters

def print_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print 'Cluster ' + str(i)
        print '----------'
        for sense, context in cluster:
            print sense + ': ' + ', '.join('(' + word + ', ' + pos + ')' for word, pos in context)

def main(file_name, model):
    print 'Processing ' + file_name
    target_word, pos = file_name.split('-', 1)
    target_word = target_word.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        target = "xyz"
        pos = "noun"

    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE)
    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append([word for word, _ in ctx])

    N, word_counts_dict = word_stats_lists(raw_ctxes)
    word_counts_list = []
    for word, count in word_counts_dict.iteritems():
        if word in model:
            word_counts_list.append((word, count))

    top_n = sorted(word_counts_list, reverse=True, key=lambda x: x[1])[:TOP_N]
    print '\n'.join('(' + word + ', ' + str(count) + ')' for word, count in top_n)

    word_embeddings = [model[word] for word, _ in top_n]

    pref = np.max(pdist(word_embeddings))
    pref = float(-1) * pref * pref
    ap = AffinityPropagation(damping=0.5, convergence_iter=15, max_iter=300, preference=pref).fit(word_embeddings)

    clusters = get_clusters(ap, [word for word, _ in top_n])

    for cluster in clusters:
        print "Cluster"
        print "-------"
        print '\n'.join(cluster)

#    os.system('rm senseclusters_scorer/answers*; rm senseclusters_scorer/key*')
#    make_answers(ap, target_word, pos)
#    make_key(file_name, target_word, pos)
#    os.system('cd senseclusters_scorer; ./senseclusters_scorer.sh answers key; cd ..')
#    os.system('cat senseclusters_scorer/report.out')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python def_gen.py <input-dir>"
        sys.exit(1)

    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    #model = None
    print "Model loaded"

    for f_name in os.listdir(sys.argv[1]):
        try:
    	    main(sys.argv[1].rstrip('/') + '/' + f_name, model)
        except Exception:
            traceback.print_exc()

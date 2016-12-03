import sys, re, nltk, os, math, traceback
from src.file_processing import *
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import numpy as np
from src.DefinitionGeneration import Definition

model_file = 'models/Google'
WINDOW_SIZE = 10
MIN_VOC_FREQ = 0.001
WORD_VEC_SIZE = 3

"""
Convert the contexts in a sense2val file into a tokenized list. Also returns the senses associated with each context.
"""
def tokenize_ctxes(file_name, target_word, stopwords, window_size, conflate_word1=None, conflate_word2=None):
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
            elif conflate_word1 and conflate_word1 in token:
                continue
            elif conflate_word2 and conflate_word2 in token:
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

def get_clusters(cluster, senses, ctxes):
    # each item is a list of contexts representing a cluster
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
        for sense, context in cluster:
            print sense + ': ' + ', '.join('(' + word + ', ' + pos + ')' for word, pos in context)

def main(file_name, model):
    print 'Processing ' + file_name
    target_word, pos, conflate_word1, conflate_word2 = do_filename(file_name)
    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE, conflate_word1, conflate_word2)

    # Only use nounds and adjs
    # filtering out certain POSes should be done here on the pos_ctxes
    #for i in range(len(pos_ctxes)):
        #pos_ctxes[i] = filter(lambda x: "NOUN" in x[1] or "ADJ" in x[1], pos_ctxes[i])

    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append([word for word, _ in ctx])

    N, word_counts_dict = word_stats_lists(raw_ctxes)
    word_counts_list = []
    for word, count in word_counts_dict.iteritems():
        word_counts_list.append((word, count))
    
    #count_thresh = math.floor(float(N)*MIN_VOC_FREQ)

    #print "Count thresh: " + str(float(N)*0.001)
    #print '\n'.join('(' + word + ', ' + str(count) + ')' for word, count in sorted(word_counts_list, reverse=True, key=lambda x: x[1]))

    # sort the words by count
    count_sorted_raw_ctxes = []
    count_sorted_pos_ctxes = []
    for i, ctx in enumerate(raw_ctxes):
        count_sorted_raw_ctxes.append(sorted(ctx, reverse=True, key=lambda x: word_counts_dict[x]))
        count_sorted_pos_ctxes.append(sorted(pos_ctxes[i], reverse=True, key=lambda x: word_counts_dict[x[0]]))

    # get the top words by count or take the top WORD_VEC_SIZE words
    final_ctxes = []
    final_ctxes_pos = []
    for i, ctx in enumerate(count_sorted_raw_ctxes):
        if len(ctx) > 0:
            final_ctxes.append(filter(lambda x: x in model, ctx)[:WORD_VEC_SIZE])
            final_ctxes_pos.append(filter(lambda x: x in model, count_sorted_pos_ctxes[i])[:WORD_VEC_SIZE])
        else:
            final_ctxes.append([])
            final_ctxes_pos.append([])

    ctx_vecs = make_context_vecs(final_ctxes, model)
    pref = np.max(pdist(ctx_vecs))
    pref = float(-1) * pref * pref
    ap = AffinityPropagation(damping=0.5, convergence_iter=15, max_iter=300, preference=pref).fit(ctx_vecs)

    clusters = get_clusters(ap, senses, final_ctxes_pos)

    print_clusters(clusters)
    return

#    definition = Definition(model)
#    for i, cluster in enumerate(clusters):
#        word_counts = defaultdict(int)
#        for _, context in cluster:
#            # word is the (word, pos) tuple
#            for word in context:
#                word_counts[word] += 1
#        word_counts_list = []
#        for (word, pos), count in word_counts.iteritems():
#            word_counts_list.append((word, pos, count))
#        word_counts_list = sorted(word_counts_list, key=lambda x: x[2], reverse=True)
#        # print 'Cluster ' + str(i)
#        # print '---------'
#        # print '\n'.join(word + ' ' + pos + ' ' + str(count) for word, pos, count in word_counts_list)
#        defPhrase = definition.process(word_counts_list)
#        print defPhrase

    os.system('rm senseclusters_scorer/answers*; rm senseclusters_scorer/key*')
    make_answers(ap, target_word, pos)
    make_key(file_name, target_word, pos)
    os.system('cd senseclusters_scorer; ./senseclusters_scorer.sh answers key; cd ..')
    os.system('cat senseclusters_scorer/report.out')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python def_gen.py <input-dir>"
        sys.exit(1)

    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    #model = None
    print "Model loaded"

    try:
    	main(sys.argv[1].rstrip('/'), model)
    except Exception:
        traceback.print_exc()
#    for f_name in os.listdir(sys.argv[1]):
#        try:
#    	    main(sys.argv[1].rstrip('/') + '/' + f_name, model)
#        except Exception:
#            traceback.print_exc()

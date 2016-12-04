import sys, os, math, traceback
from src.file_processing import *
from src.tokenizing import *
from src.cluster import *
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import pdist
import numpy as np
from src.DefinitionGeneration import Definition

model_file = 'models/Google'
WINDOW_SIZE = 10
MIN_VOC_FREQ = 0.001
WORD_VEC_SIZE = 5

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

def cluster_one(file_name, model):
    print 'Processing ' + file_name
    target_word, pos, conflate_word1, conflate_word2 = do_filename(file_name)
    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE, conflate_word1, conflate_word2)

    # Only use nounds and adjs
    # filtering out certain POSes should be done here on the pos_ctxes
    # in general this seems to mess things up
    #for i in range(len(pos_ctxes)):
        #pos_ctxes[i] = filter(lambda x: "NOUN" in x[1] or "ADJ" in x[1], pos_ctxes[i])

    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append([word for word, _ in ctx])

    N, word_counts_dict = word_stats_lists(raw_ctxes)
    word_counts_list = []
    for word, count in word_counts_dict.iteritems():
        word_counts_list.append((word, count))
    
    count_thresh = int(math.ceil(float(N)*MIN_VOC_FREQ))

    print "Count thresh: " + str(count_thresh)
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
            final_ctxes.append(filter(lambda x: x in model and word_counts_dict[x] >= count_thresh, ctx)[:WORD_VEC_SIZE])
            final_ctxes_pos.append(filter(lambda x: x in model and word_counts_dict[x] >= count_thresh, count_sorted_pos_ctxes[i])[:WORD_VEC_SIZE])
        else:
            final_ctxes.append([])
            final_ctxes_pos.append([])

    ctx_vecs = make_context_vecs(final_ctxes, model)
    pref = np.max(pdist(ctx_vecs))
    pref = float(-1) * pref * pref
    print pref
    ap = AffinityPropagation(damping=0.75, convergence_iter=50, max_iter=1000, preference=-40).fit(ctx_vecs)

    clusters = get_clusters(ap, senses, final_ctxes_pos)

    #print_clusters(clusters)

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
        print "Error: takes one argument"
        print "usage: python def_gen.py <sense2val file>"
        sys.exit(1)

    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    print "Model loaded"

    try:
    	cluster_one(sys.argv[1].rstrip('/'), model)
    except Exception:
        traceback.print_exc()

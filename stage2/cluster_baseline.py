import sys, os, math, traceback, copy
from src.file_processing import *
from src.tokenizing import *
from src.cluster import *
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from scipy.spatial.distance import pdist
import numpy as np
from src.DefinitionGeneration import Definition

model_file = 'models/Google'
WINDOW_SIZE = 10
MIN_VOC_FREQ = 0.001
WORD_VEC_SIZE = 15
PREF_INC = 0.25

def get_clusters(cluster, senses):
    # each item is a list of contexts representing a cluster
    clusters = []
    # cluster.label -> index in clusters
    clusterlabel_to_index = {}
    for i, label in enumerate(cluster.labels_):
        if label not in clusterlabel_to_index:
            clusterlabel_to_index[label] = len(clusters)
            clusters.append([])
        clusters[clusterlabel_to_index[label]].append(senses[i])
    return clusters

def print_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print 'Cluster ' + str(i)
        print '----------'
        for sense, context in cluster:
            print sense + ': ' + ', '.join('(' + word + ', ' + pos + ')' for word, pos in context)

"""
pref value is if we want to hard code a preference value. Its primary purpose is for
debugging/fine-tuning.
"""
def cluster_tfidfs(file_name, model, pref=None):
    print 'Processing ' + file_name
    target_word, pos, conflate_word1, conflate_word2 = do_filename(file_name)
    if conflate_word1 and conflate_word2:
        return
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
    
    # ctxes_tfidf is already sorted by tfidf
    ctxes_tfidf = tf_idfs(raw_ctxes)
    
    for i in range(len(ctxes_tfidf)):
        ctxes_tfidf[i] = filter(lambda x: x[0] in model, ctxes_tfidf[i])

    #for ctx in ctxes_tfidf:
        #print '[' + ', '.join('(' + word + ', ' + str(tf_idf) + ')' for word, tf_idf in ctx[:WORD_VEC_SIZE])

    final_ctxes = []
    for ctx in ctxes_tfidf:
        final_ctxes.append(ctx[:WORD_VEC_SIZE])

    ctx_vecs = make_context_vecs_tfidf(final_ctxes, model)
    prefs = []
    if not pref:
        pref = np.max(pdist(ctx_vecs))
        pref = float(-1) * pref * pref
        inc = PREF_INC * pref
        for i in range(-5, 5):
            prefs.append(inc*float(i) + pref)
    else:
        # debugging
        prefs.append(pref)

    
    ch_scores = []
    all_clusters = []
    for ferp in prefs:
        print ferp
        ap = AffinityPropagation(damping=0.75, convergence_iter=50, max_iter=1000, preference=ferp).fit(ctx_vecs)
    
        clusters = get_clusters(ap, senses)
        if len(clusters) != 1 and len(clusters) != len(ctx_vecs):
            ch_score = metrics.calinski_harabaz_score(ctx_vecs, ap.labels_)
        else:
            # scoring metrics can't operate on one cluster
            ch_score = 0
        ch_scores.append(ch_score)
        all_clusters.append(copy.copy(ap.labels_))

    best_cluster = all_clusters[ch_scores.index(max(ch_scores))]
            
    os.system('rm senseclusters_scorer/answers*; rm senseclusters_scorer/key*')
    make_answers(best_cluster, target_word, pos)
    make_key(file_name, target_word, pos)
    os.system('cd senseclusters_scorer; ./senseclusters_scorer.sh answers key; cd ..')
    os.system('cat senseclusters_scorer/report.out')
    

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
    for i in range(len(pos_ctxes)):
        pos_ctxes[i] = filter(lambda x: "NOUN" in x[1] or "ADJ" in x[1], pos_ctxes[i])

    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append([word for word, _ in ctx])

    N, word_counts_dict, _ = word_stats_lists(raw_ctxes)
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
    os.system('echo "Calculated pref value: ' + str(pref) + '"')

    for ferp in [-5, -5.5, -6, -6.5, -7, -7.5, -8]:
        ap = AffinityPropagation(damping=0.75, convergence_iter=40, max_iter=300, preference=ferp).fit(ctx_vecs)

        clusters = get_clusters(ap, senses)

        ch_score = metrics.calinski_harabaz_score(ctx_vecs, ap.labels_)
        sh_score = metrics.silhouette_score(ctx_vecs, ap.labels_, metric='euclidean')

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
        os.system('echo "ch: ' + str(ch_score) + ', sh: ' + str(sh_score) + '"')
        os.system('cat senseclusters_scorer/report.out')

if __name__ == '__main__':

    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    #model = Word2Vec.load('models/3_parts_of_wiki')
    print "Model loaded"

    try:
        for file_name in os.listdir('input'):
            cluster_tfidfs('input/' + file_name, model)
    except Exception:
        traceback.print_exc()

import sys, os, math, traceback, copy, random
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
MAX_CLUSTERS = 8
PRIMING_SIZE = 300

def get_clusters(cluster):
    # each item is a list of contexts representing a cluster
    clusters = []
    # cluster.label -> index in clusters
    clusterlabel_to_index = {}
    for i, label in enumerate(cluster):
        if label not in clusterlabel_to_index:
            clusterlabel_to_index[label] = len(clusters)
            clusters.append([])
        clusters[clusterlabel_to_index[label]].append(i)
    return clusters

def score_cluster(cluster, answer_file, target_word, pos):
    os.system('rm senseclusters_scorer/answers*; rm senseclusters_scorer/key*')
    make_answers(cluster, target_word, pos)
    make_key(answer_file, target_word, pos)
    os.system('cd senseclusters_scorer; ./senseclusters_scorer.sh answers key; cd ..')
    os.system('cat senseclusters_scorer/report.out')

"""
pref value is if we want to hard code a preference value. Its primary purpose is for
debugging/fine-tuning.
"""
def cluster_tfidfs(file_name, model, pref=None):
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
    
    # ctxes_tfidf is already sorted by tfidf
    ctxes_tfidf = tf_idfs(raw_ctxes)
    
    for i in range(len(ctxes_tfidf)):
        ctxes_tfidf[i] = filter(lambda x: x[0] in model, ctxes_tfidf[i])

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
    
    if len(ctx_vecs) > PRIMING_SIZE:
        # list index -> instance id
        li_to_ii = random.sample(range(len(ctx_vecs)), PRIMING_SIZE)
        priming_sample = [ctx_vecs[ii] for ii in li_to_ii]
    else:
        priming_sample = ctx_vecs

    ch_scores = []
    all_clusters = []
    for ferp in prefs:
        ap = AffinityPropagation(damping=0.75, convergence_iter=50, max_iter=1000, preference=ferp).fit(priming_sample)
    
        num_clusters = len(set(ap.labels_))
        if num_clusters > MAX_CLUSTERS:
            continue
        if num_clusters != 1 and num_clusters != len(priming_sample):
            ch_score = metrics.calinski_harabaz_score(priming_sample, ap.labels_)
        else:
            # scoring metrics can't operate on one cluster
            ch_score = 0
        ch_scores.append(ch_score)
        all_clusters.append(copy.copy(ap.labels_))

    best_cluster = all_clusters[ch_scores.index(max(ch_scores))]
    instance_clusters = get_clusters(best_cluster)
    final_cluster = []
    if len(priming_sample) != len(ctx_vecs):
        # keep track of the contexts we clustered already
        # index in ctx_vecs -> cluster label
        assigned = {ii:best_cluster[li] for li, ii in enumerate(best_cluster)}
        # make a cluster vector for each cluster
        cluster_vecs = []
        for ctx_ids in instance_clusters:
            cluster_vecs.append(sum([priming_sample[i] for i in ctx_ids]) / float(len(ctx_ids)))
        # assign the rest of our contexts
        for i, vec in enumerate(ctx_vecs):
            if i in assigned:
                final_cluster.append(assigned[i])
            else:
                best_match = -1
                max_sim = -2.0
                for i, cluster_vec in enumerate(cluster_vecs):
                    sim = np.dot(vec, cluster_vec)/(np.linalg.norm(vec)* np.linalg.norm(cluster_vec))
                    if sim > max_sim:
                        max_sim = sim
                        best_match = i
                final_cluster.append(best_match)
    else:
        final_cluster = best_cluster

    score_cluster(final_cluster, file_name, target_word, pos)

    final_clusters = get_clusters(final_cluster)
    defgen_model = Word2Vec.load('models/3_parts_of_wiki_lowercase')
    definition = Definition(defgen_model, pos)
    for i, cluster in enumerate(final_clusters):
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        for instance_id in cluster:
            # word is the (word, pos) tuple
            seen = set()
            for word in pos_ctxes[instance_id]:
                if word[0] in defgen_model:
                    word_counts[word] += 1
                    seen.add(word)
            for word in seen:
                doc_counts[word] += 1
        word_counts_list = []
        for (word, pos), count in word_counts.iteritems():
            word_counts_list.append((word, pos, count))
        word_counts_list = sorted(word_counts_list, key=lambda x: x[2], reverse=True)
        defPhrase = definition.process(word_counts_list, doc_counts)
        print defPhrase
    sys.stdout.flush()

if __name__ == '__main__':

    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    print "Model loaded"

    for file_name in os.listdir('input'):
        cluster_tfidfs('input/' + file_name, model)

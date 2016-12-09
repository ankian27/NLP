"""
Author: Brandon Paulsen
"""

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

# The model file used for cluster
model_file = 'models/Google'
# The number of words to the left and right of the target word to
# consider in clustering
WINDOW_SIZE = 10
# The maximum number of words we will include in a context vector
WORD_VEC_SIZE = 15
# The amount we increment the affinity propogation's preference value
# for each trial
PREF_INC = 0.25
# The maximum number of clusters
MAX_CLUSTERS = 8
# The number of priming instances to use
PRIMING_SIZE = 300


"""
Converts a labels list to groups of instance ids which represent clusters.
@param cluster: a list of labels, like the ones generated by any of sklearn's clustering objects
@return: a list of lists of instance ids representing the clusters
"""
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


"""
A short function to score our clusters.
@param cluster: a list of integer labels corresponding to the assigned cluster for each instance
@param answer_file: the original sense2val file that clustering was performed on
@param target_word: the target word
@param pos: the part of speech of the target word
"""
def score_cluster(cluster, answer_file, target_word, pos):
    os.system('rm senseclusters_scorer/answers*; rm senseclusters_scorer/key*')
    make_answers(cluster, target_word, pos)
    make_key(answer_file, target_word, pos)
    os.system('cd senseclusters_scorer; ./senseclusters_scorer.sh answers key; cd ..')
    os.system('cat senseclusters_scorer/report.out')

"""
The main clustering function. Takes the path to a single sense2val file and outputs clusters
in sensecluster_scorer/. Clustering is done using affinity propogation, and a Word2Vec model.
The Word2Vec model is used to calculate similarity scores between separate instances. Affinity
propogation takes these similarities and creates clusters based on them.
@param file_name: string path to the sense2val input file
@param model: A trained Word2Vec model
@param pref: hard-coded prefernce value for affinity propogation. Only used for debugging/fine-tuning.
"""
def cluster_tfidfs(file_name, model, pref=None):
    target_word, pos, conflate_word1, conflate_word2 = do_filename(file_name)
    stopwords = set(line.strip() for line in open('stopwords.txt', 'r'))

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE, conflate_word1, conflate_word2)

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
    #defgen_model = Word2Vec.load('models/3_parts_of_wiki_lowercase')
    for i, cluster in enumerate(final_clusters):
        definition = Definition(model, pos)
        #print "Cluster " + str(i)
        #print "----------"
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        for instance_id in cluster:
            # word is the (word, pos) tuple
            seen = set()
            print "ID: " + str(instance_id) + " " + ' '.join('(' + word + ',' + str(tfidf) + ')' for word, tfidf in final_ctxes[instance_id])
            for word in pos_ctxes[instance_id]:
                if word[0] in model:
                    word_counts[word] += 1
                    seen.add(word)
            for word in seen:
                doc_counts[word] += 1
        word_counts_list = []
        for (word, pos), count in word_counts.iteritems():
            word_counts_list.append((word, pos, count))
        word_counts_list = sorted(word_counts_list, key=lambda x: x[2], reverse=True)
        #print "Top 10 words for cluster: "
        #print word_counts_list[:10]
        defPhrase = definition.process(word_counts_list, doc_counts)
        print defPhrase
    #print "--End--\n"

    sys.stdout.flush()

if __name__ == '__main__':
    nltk.data.path.append('/home/csugrads/pauls658/nltk_data')
    print "Loading model"
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    print "Model loaded"

    for file_name in os.listdir('input'):
        cluster_tfidfs('input/' + file_name, model)

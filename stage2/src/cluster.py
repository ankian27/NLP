from collections import defaultdict
import math

"""
ctxes should be raw ctxes
"""
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

"""
ctxes should be lists of (word, tfidf) pairs
"""
def make_context_vecs_tfidf(ctxes, model):
    vecs = []
    for ctx in ctxes:
        if len(ctx) == 0:
            vecs.append(model['the'] - model['the'])
            continue
        tfidf_sum = sum([x[1] for x in ctx])
        vec = None
        for word, tfidf in ctx:
            weight = float(float(tfidf)/tfidf_sum)
            if vec is None:
                vec = model[word]*weight
            else:
                vec += model[word]*weight
        vecs.append(vec)
    return vecs

"""
ctxes should be raw ctxes
"""
def word_stats_lists(ctxes):
    total_count = 0
    # word -> overall counts
    word_counts = defaultdict(int)
    # word -> docs the word appears in
    doc_counts = defaultdict(set)
    for i, ctx in enumerate(ctxes):
        tot_ctx_words = 0
        total_count += len(ctx)
        for word in ctx:
            doc_counts[word].add(i)
            word_counts[word] += 1
    return total_count, word_counts, doc_counts

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

"""
ctxes should be the raw ctxes
"""
def tf_idfs(ctxes):
    N, wcs, doc_counts = word_stats_lists(ctxes)
    ctxes_tfidf = []
    for ctx in ctxes:
        ctxes_tfidf.append([])
        for word in ctx:
            tf = float(math.log1p(float(wcs[word])))
            idf = float(math.log10(float(N)/float(len(doc_counts[word]))))
            ctxes_tfidf[-1].append((word, float(tf*idf)))
        ctxes_tfidf[-1].sort(reverse=True, key=lambda x: x[1])
    return ctxes_tfidf

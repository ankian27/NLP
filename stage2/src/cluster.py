

from collections import defaultdict
import math

"""
Converts a list of contexts into a list of vectors which represent the contexts. Specifically, each
context in ctxes should be a list of word tokens. Each word token should appear in the passed in model.
A context vector for a single context is the sum of all its token's vectors, normalized by the number of
tokens in the context.
@param ctxes: a list of contexts where each context is a list of string tokens
@param model: a trained Word2Vec model
@return: a list of context vectors
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
Converts a list of contexts into a list of vectors which represent the contexts. Specifically, each
context in ctxes should be a list pairs, where the first element is the word and the second is the
word's tfidf weight. The tfidf weight is used to weight the vector in the context vector. That is,
a word with a higher tfidf weight will have more influence on the resulting context vector.
@param ctxes: a list of contexts where each context is a list of pairs of the form (word, tfidf weight)
@param model: a trained Word2Vec model
@return: a list of context vectors
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
Calculates various statistics about a list of contexts. Specifically, it calculates overall word counts,
document counts for each words, and the total number of words.
@param ctxes: a list of contexts where each context is a list of words (strings)
@return: the total number of words, the word counts for each word, and the document counts for each word
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

"""
Calculates tfidf weights for a list of contexts.
@param ctxes: a list of contexts where each context is a list of words (strings)
@return: the same list of contexts, but with each word replaced by a pair which contains
         the original word and the word's tfidf weight
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

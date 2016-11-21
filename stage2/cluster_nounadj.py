import sys
import re
import math
import nltk
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN

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
        # - the target word itself
        # - tokens consisting of only punctuation
        # - tokens with no alphabetical characters
        # - tokens with a '.' in them (this indicates an acronym which
        #   is generally not useful)
        # - tokens with single quote in them (these are usually the
        #   second half of a conjuction, for example "n't" or "'ve"
        #   which are not usually useful)
        for token in nltk.word_tokenize(ctx):
            token = token.lower()
            if not token or token in stopwords or (target_word in token and token != target_word + 'target'): 
                continue 
            elif not any(c.isalpha() for c in token):
                continue
            elif '\'' in token or '.' in token:
                continue
            elif '-' in token:
                # nltk doesn't split hyphenated words for us
                for t in token.split('-'):
                    if t:
                    	tokens.append(t)
            else:
                tokens.append(token)
        # index function will throw an error if we don't find the target
        # word
        target_word_i = tokens.index(target_word + 'target')
        ctx_tokens.append(tokens[target_word_i - window_size:target_word_i] + tokens[target_word_i + 1:target_word_i + window_size + 1])
        senses.append(sense)
    return ctx_tokens, senses

def tokenize_nounadj(file_name, target_word, stopwords, window_size):
    rename_re = re.compile(r'<head>(.*?)</head>')
    rm_head_re = re.compile(r'</?head>')
    ctx_tokens = []
    senses = []
    for sense, ctx in get_ctxes(file_name):
        ctx = rename_re.sub(' ' + target_word + 'target ', ctx, 1)
        ctx = rm_head_re.sub('', ctx)
        ctx = filter(lambda c: 0 < ord(c) < 127, ctx)

        tagged_ctx = nltk.pos_tag(nltk.word_tokenize(ctx))
        simpletags_ctx = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_ctx]
        target_word_i = 0
        while simpletags_ctx[target_word_i][0] != target_word + 'target':
            target_word_i += 1

        nounadjs = set()
        for token, pos in simpletags_ctx[target_word_i - window_size:target_word_i] + simpletags_ctx[target_word_i + 1:target_word_i + window_size + 1]:
            if pos == 'NOUN' or pos == 'ADJ':
                if '-' in token:
                    for word in token.split('-'):
                        nounadjs.add(word.lower())
                else:
                    nounadjs.add(token.lower())
        ctx_tokens.append(nounadjs)
        senses.append(sense)

    return ctx_tokens, senses

def word_stats(ctxes):
    total_count = 0
    word_counts = defaultdict(int)
    ctx_word_counts = []
    for ctx in ctxes:
        tot_ctx_words = 0
        for word, count in ctx.iteritems():
            total_count += count
            word_counts[word] += count
            tot_ctx_words += count
        ctx_word_counts.append(count)
    return total_count, word_counts, ctx_word_counts

def calc_ppmis(ctxes, N, word_counts, ctx_counts):
    ctx_pmis = []
    N = float(N)
    for i, ctx in enumerate(ctxes):
        ctx_pmi = []
        for word, count in ctx.iteritems():
            if word_counts[word] < MIN_VOC_FREQ:
                continue
            p_xy = float(float(count)/N)
            p_x = float(float(word_counts[word])/N)
            p_y = float(float(ctx_counts[i])/N)
            pmi = float(p_xy/(p_x*p_y))
            ctx_pmi.append((word, pmi))
        ctx_pmis.append(sorted(ctx_pmi, key=lambda x: x[1], reverse=True))
    return ctx_pmis

def calc_distances(ctxes, model):
    distances = []
    for ctx1 in ctxes:
        distances.append([])
        for ctx2 in ctxes:
            if isinstance(ctx2, str):
                distances[-1].append(1 - model.similarity(ctx1, ctx2))
            else:
                distances[-1].append(1 - model.n_similarity(ctx1, ctx2))
    return distances


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
    ctxes, senses = tokenize_nounadj(file_name, target_word, stopwords, WINDOW_SIZE)
    for i, ctx in enumerate(ctxes):
        ctxes[i] = filter(lambda x: x in model, ctx)
        #print 'context ' + str(i) + ' ' + ' '.join(ctx)

#    ctxes_bow = []
#    for ctx in ctxes:
#        bow = defaultdict(int)
#        for word in ctx:
#            bow[word] += 1
#        ctxes_bow.append(bow)

    #for i, context in enumerate(ctxes):
        #print 'Context ' + str(i) + '| sense: ' + senses[i] + ' | words: ' + ' '.join(context)

    #N, word_counts, ctx_counts = word_stats(ctxes_bow)
    #ctx_pmis = calc_ppmis(ctxes_bow, N, word_counts, ctx_counts)

    #wcs_list = []
    #for word, count in word_counts.iteritems():
        #wcs_list.append((word, count))
    #wcs_list = sorted(wcs_list, key=lambda x: x[1], reverse=True)
    #vocab = [x[0] for x in filter(lambda x: x[0] in model, wcs_list)]
    #print '\n'.join(word + ': ' + str(count) for word, count in wcs_list)
    
    distances = calc_distances(ctxes, model)
    
    db = DBSCAN(eps=0.5, min_samples=10, metric='precomputed', n_jobs=-1).fit(distances)

    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        clusters[label].append(senses[i])
    for label, ctxes in clusters.iteritems():
        print str(label)
        print ' '.join(ctxes)

    return

    ctx_vecs = []
    for ctx in ctx_pmis:
        ctx_vecs.append([word for word, _ in ctx[:WORD_VEC_SIZE]])
        print ctx_vecs[-1]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python def_gen.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

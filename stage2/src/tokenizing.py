"""
Author: Brandon Paulsen
"""

import nltk, re
from file_processing import *

"""
Convert the contexts in a sense2val file into a tokenized list. Also returns the senses associated with each context for debugging/development.
This functions performs several pre-processing steps. Specifically, it removes any tokens that have the target word or conflate words in them.
This is the purpose of taking the target word and the conflate words. It also removes the stopwords specified by the stopwords param. In
addition, we do the POS tagging in this step while we have the complete contexts. 
@param file_name: a string path to a sense2val file
@param target_word: the word being disambiguated
@param stopwords: a set or list of stopwords
@param window_size: the number of context words before and after to include
@param conflate_word1: the first conflated word
@param conflate_word2: the second conflated word
@return: a list of tokenized contexts where each context is a list of pairs of the form (token, POS)
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


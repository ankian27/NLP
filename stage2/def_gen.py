import sys
import re
import nltk
from collections import defaultdict

WINDOW_SIZE = 10

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
        # - tokens consisting of only punctuation
        # - tokens with no alphabetical characters
        # - tokens with a '.' in them (this indicates an acronym which
        #   is generally not useful)
        # - tokens with single quote in them (these are usually the
        #   second half of a conjuction, for example "n't" or "'ve"
        #   which are not usually useful)
        for token in nltk.word_tokenize(ctx):
            token = token.lower()
            if not token or token in stopwords: 
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

    # ctxes is a list of lists of tokens
    # senses is a list sense strings, where sense[i] is the sense of
    # ctxes[i]
    ctxes, senses = tokenize_ctxes(file_name, target_word, stopwords, WINDOW_SIZE)

    for i, context in enumerate(ctxes):
        print 'Context ' + str(i) + '| sense: ' + senses[i] + ' | words: ' + ' '.join(context)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python def_gen.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

import string
import nltk
import re

"""
Contains the utility functions for processing contexts pulled from
senseval2 formatted files.


"""

# read the stopwords into set
sws = set(line.strip() for line in open('stopwords.txt', 'r'))

# used to the split the raw context into two seperate sections: the
# context before the target word, and the context after the target word
split_re = re.compile(r'<head>(.*?)</head>')
rm_head_re = re.compile(r'</?head>')

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

"""
FUN2
Tokenizes the given context, performing any special processing
on the file, specifically removing stopwords and converting all the words
to lower case. In addition, we remove non-ascii characters from the contexts
because they can break hdp-wsi, and we strip any head tags from the context.
@param context: a raw context pulled a senseval2 formatted xml file
@param ctx_len: the number of tokens on each side of the target to include in
                the returned tokenized context
@param target_word: the target word for this context. hdp-wsi relies on the
                    exact target word appearing in the context (without
                    any affixes), so we replace the word between the
                    <head> tags with the canonical target_word.
@return: a list of tokens representing the tokenized context
"""
def tokenize(context, ctx_len, target_word):
    context = strip_non_ascii(context)
    b4, target, after = split_re.split(context, 1)
    after = rm_head_re.sub('', after)
    doc = []
    tmp1 = []
    tmp2 = []
    for word in nltk.word_tokenize(b4):
        # nltk doesn't split hyphenated words for us
        if '-' in word:
            for word1 in word.split('-'):
                    tmp1.append(word1)
        else:
            tmp1.append(word)
    for word in nltk.word_tokenize(after):
        # nltk doesn't split hyphenated words for us
        if '-' in word:
            for word1 in word.split('-'):
                    tmp2.append(word1)
        else:
            tmp2.append(word)

    for word in reversed(tmp1):
        word = word.lower()
        if word not in sws and word.isalpha():
            doc.append(word)
        if len(doc) >= ctx_len:
            break
    doc.reverse()
    doc.append(target_word.lower())
    for word in tmp2:
        word = word.lower()
        if word not in sws and word.isalpha():
            doc.append(word)
        if len(doc) >= ctx_len*2:
            break
        
    return doc

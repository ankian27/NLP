import nltk
import gensim
import re
import string

class SentenceIter():
    def __init__(self, f_name, tar_word):
        self.file_name = f_name
        self.target_word = tar_word

    def __iter__(self):
        for sentence in self.get_sentences(self.file_name, self.target_word):
            yield sentence

    def getCtxes(self, f):
        #RE1
        ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
        with open(f, 'rb') as f_ref:
            buf = f_ref.read()
        for sense, ctx in ctx_re.findall(buf):
            yield sense, ctx.strip()
    
    def get_sentences(self, file_name, target_word):
        rename_re = re.compile(r'<head>(.*?)</head>')
        rm_head_re = re.compile(r'</?head>')
        tar_id = 0
        for senese, ctx in self.getCtxes(file_name):
            ctx = rename_re.sub(r' \1_' + str(tar_id) + ' ', ctx, 1)
            ctx = rm_head_re.sub('', ctx)
            ctx = filter(lambda c: 0 < ord(c) < 127, ctx)
            sentence = []
            for token in nltk.word_tokenize(ctx):
                if not token: 
                    continue 
                elif token == '.':
                    yield sentence
                    sentence = []
                elif not any(c.isalpha() for c in token):
                    continue
                elif '-' in token:
                    # nltk doesn't split hyphenated words for us
                    sentence.extend([t.lower() for t in token.split('-')])
                else:
                    sentence.append(token.lower())
            tar_id += 1
    
class Word2VecModel():
    """
    constructs a word2vec model given a sense2val formatted file.
    """
    def __init__(self, file_name, target_word, pos):
        self.model = gensim.models.Word2Vec(SentenceIter(file_name, target_word))

from src.corp import Corp
from src.corp import tokenize
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities.docsim import Similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict
import xmltodict
import optparse
import sys
import re

def getCtxes(f):
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

"""
Disambiguates the given sense2eval xml file, f. The target parameter is only necessary if
the target word isn't marked with <head> tags.
@param f: the file path to the sense2eval xml formatted file
@param target: optional target word
"""
def disam(f, target=None):
    remove_head_re = re.compile(r'</?head>')
    target, pos  = f.split('-', 1)
    with open(target + "." + pos[0] + ".lemma", 'w+') as f_ref:
        for sense, ctx in getCtxes(f):
            f_ref.write(remove_head_re.sub('', ctx) + '\n')

if __name__ == '__main__':
    parser = optparse.OptionParser(description='Disambiguate a target word.')
    parser.add_option('-t', dest='target', action='store',
                        default=None,
                        help='the target word')
    # first arg is the program name. Ignore it
    (options, f) = parser.parse_args(sys.argv[1:])
    if not f:
        print "need an xml file to parse"
        sys.exit(1)
    disam(f[0], options.target)


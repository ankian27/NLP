from src.corp import tokenize
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus
from collections import defaultdict
import sys
import re

def getCtxes(f):
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

def writeCorpToHDPWSI(f, corp):
    target, pos = f.split('-', 1)
    target = target.split('/')[-1]
    num_ctxes = 0
    with open("hdp-wsi/wsi_input/example/all/" + target + "." + pos[0] + ".lemma", 'w+') as f_ref:
        for doc in corp:
            num_ctxes += 1
            f_ref.write(' '.join(word for word in doc) + "\n")
            #hdp format
            #print str(len(doc)) + " " + ' '.join(str(term) + ":" + str(count) for term, count in doc)

    with open("hdp-wsi/wsi_input/example/num_test_instances.all.txt", 'w+') as f_ref:
        f_ref.write(target + "." + pos[0] + " " + str(num_ctxes))


def writeCtxesToHDPWSI(f):
    target, pos = f.split('-', 1)
    target = target.split('/')[-1]
    remove_head_re = re.compile(r'</?head>')
    num_ctxes = 0
    with open("hdp-wsi/wsi_input/example/all/" + target + "." + pos[0] + ".lemma", 'w+') as f_ref:
        for sense, ctx in getCtxes(f):
            num_ctxes += 1
            f_ref.write(remove_head_re.sub('', ctx).strip() + "\n")

    with open("hdp-wsi/wsi_input/example/num_test_instances.all.txt", 'w+') as f_ref:
        f_ref.write(target + "." + pos[0] + " " + str(num_ctxes))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "usage: python writeToHDP.py <senseval2-xml-file>"
        sys.exit(1)
    
    #writeCtxesToHDPWSI(sys.argv[1])
    #sys.exit(1)

    corp = []
    for sense, ctx in getCtxes(sys.argv[1]):
        corp.append(tokenize(ctx, 500))
        #print ' '.join(word for word in corp[-1])
    writeCorpToHDPWSI(sys.argv[1], corp)

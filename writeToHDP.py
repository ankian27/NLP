from src.corp import tokenize
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus
from collections import defaultdict
import sys
import re

"""
A script to convert senseval2 xml formatted input file to a format that
the hdp-wsi tool can process. Any special filtering on the input contexts
is done here as well, because the hdp-wsi tool doesn't remove stopwords,
convert to lowercase, etc...
The filtering is done by the tokenize() function (the first import line)
which is defined in src/corp.py

@AUTHOR: Brandon Paulsen
"""

"""
FUN1
A helper function which parses the given senseval2 formatted file without having to
read the entire file into memory.
@param f: a string which is the file path to the senseval2 formatted file
@return: a generator which will return a tuple when called. The tuple has
         the form (sense_id, context)
"""
def getCtxes(f):
    #RE1
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

"""
FUN3
Writes a given corpus into the hdp-wsi tool's input folder in a format that the tool
can understand. 
@param f: the senseval2 file we initially read from. This parameter is only used to 
          get the target word and POS. 
@param corp: a list of lists which represent our corpus. Each list in corp should be
             the tokenized form of the context. For example, a single list item in corp
             would look like:
             ['the', 'dog', 'barked', 'at', 'the', 'mouse']
             Note that any preprocessing should already be done at this point (eg. removal
             of stopwords or case-conversion)
"""
def writeCorpToHDPWSI(f, corp):
    target, pos = f.split('-', 1)
    target = target.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        target = "xyz"
        pos = "noun"
    num_ctxes = 0
    with open("hdp-wsi/wsi_input/example/all/" + target + "." + pos[0] + ".lemma", 'w+') as f_ref:
        for doc in corp:
            num_ctxes += 1
            f_ref.write(' '.join(word for word in doc) + "\n")
            #hdp format
            #print str(len(doc)) + " " + ' '.join(str(term) + ":" + str(count) for term, count in doc)

    with open("hdp-wsi/wsi_input/example/num_test_instances.all.txt", 'w+') as f_ref:
        f_ref.write(target + "." + pos[0] + " " + str(num_ctxes))


"""
A simplified version of the above function for when we don't want to do any preprocessing
to the contexts of the senseval2 formatted file. This function takes the senseval2 file
and writes it to the hdp-wsi tool's input directory in the format that the tool expects. We
don't use this function currently, but we didn't want to get rid of it becasue of brandon is
emotionally attached to the code he writes.
@param f: a string which is the file path to the senseval2 file
"""
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
    f = sys.argv[1]
    # COM1
    # Get the target word from the file name
    target, pos = f.split('-', 1)
    target = target.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        target = "xyz"
        pos = "noun"
    # COM3
    # Convert input file into list of tokenized documents
    # Each document is converted into a list of tokens, so corp ends up
    # looking something like this:
    # [
    #  ["hello", "world"],
    #  ["my", "name", "is", "brandon"],
    #  ...
    # ]
    corp = []
    for sense, ctx in getCtxes(sys.argv[1]):
        corp.append(tokenize(ctx, 500, target))
        #print ' '.join(word for word in corp[-1])
    # Write the tokenized document to the hdp-wsi input directory
    writeCorpToHDPWSI(sys.argv[1], corp)

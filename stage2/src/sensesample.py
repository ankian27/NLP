import sys
import random
from src.file_processing import *
from src.tokenizing import *
from src.cluster import *

WINDOW_SIZE = 10
TOP_N = 50
WORD_VEC_SIZE = 4

def sensesample_random(file_name, sample_size):
    target_word, pos, conf_word1, conf_word2 = do_filename(file_name) 

    header, instances, trailer = get_instances(file_name)
    
    if conf_word1 and conf_word2:
        SAMPLE_FILE = conf_word1 + '-' + conf_word2 + '-sample.xml'
    else:
        SAMPLE_FILE = target_word + '-' + pos + '-sample.xml'

    num_instances = 0
    with open(SAMPLE_FILE, 'wb+') as fd:
        fd.write(header)
        # write out the instances that we want in the sample
        written = set()
        while len(written) < sample_size:
            i = random.randint(0, len(instances) - 1)
            if i not in written:
                fd.write(instances[i] + '\n')
                written.add(i)

def sensesample_wcs(file_name):
    sws = set(word.strip() for word in open('stopwords.txt', 'r'))
    target_word, pos, conf_word1, conf_word2 = do_filename(file_name) 
    pos_ctxes, senses = tokenize_ctxes(file_name, target_word, sws, WINDOW_SIZE, conf_word1, conf_word2)

    raw_ctxes = []
    for ctx in pos_ctxes:
        raw_ctxes.append(set(word for word, pos in ctx))
    # raw_ctxes contains BOW's now

    #figure out which contexts we want to use
    N, wcs_dict = word_stats_lists(raw_ctxes)
    wcs_list = [(word, count) for word, count in wcs_dict.iteritems()]
    wcs_list = sorted(wcs_list, reverse=True, key=lambda x: x[1])
    top_n = set(word for word, count in wcs_list[:TOP_N])
    
    want_instances = set()
    for i, ctx in enumerate(raw_ctxes):
        if len(ctx & top_n) >= WORD_VEC_SIZE:
            want_instances.add(i)

    #get the instances
    header, instances, trailer = get_instances(file_name)
    
    if conf_word1 and conf_word2:
        SAMPLE_FILE = conf_word1 + '-' + conf_word2 + '-sample.xml'
    else:
        SAMPLE_FILE = target_word + '-' + pos + '-sample.xml'

    num_instances = 0
    with open(SAMPLE_FILE, 'wb+') as fd:
        fd.write(header)
        # write out the instances that we want in the sample
        for i, instance in enumerate(instances):
            if i in want_instances:
                num_instances += 1
                fd.write(instances[i] + '\n')
        fd.write(trailer)
    print "Wrote " + str(num_instances) + " instance to sample file"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Error: script takes one sense2val file as an argument"
        print "Usage: python sensesample.py <sense2val-file>"
        exit(1)

    reload(sys)
    sys.setdefaultencoding('utf-8')
    sensesample_random(sys.argv[1], 200)

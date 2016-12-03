import sys
import random
from src.file_processing import *

def sensesample(file_name, sample_size):
    target_word, pos, conf_word1, conf_word2 = do_filename(file_name) 

    header, instances, trailer = get_instances(file_name)
    
    if conf_word1 and conf_word2:
        SAMPLE_FILE = conf_word1 + '-' + conf_word2 + '-sample.xml'
    else:
        SAMPLE_FILE = target_word + '-' + pos + '-sample.xml'

    with open(SAMPLE_FILE, 'wb+') as fd:
        fd.write(header)
        for instance_id in random.sample(range(len(instances)), sample_size):
            fd.write(instances[instance_id] + '\n')
        fd.write(trailer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Error: script takes one sense2val file as an argument"
        print "Usage: python sensesub.py <sense2val-file>"
        exit(1)

    reload(sys)
    sys.setdefaultencoding('utf-8')
    main(sys.argv[1], 100)

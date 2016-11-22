from src.DefinitionGeneration import Definition
from collections import defaultdict
import sys
import re
from nltk.tag import pos_tag, map_tag
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer

"""
Performs post processing on the results of the hdp-wsi tool. This includes parsing hdp-wsi's
output and generating definitions. In addition, we experimented with absorbing the smaller
clusters into the larger clusters because the hdp-wsi consistently produced many more
senses than there actually were, but this actually hurt precision for reasons we don't
understand.

@AUTHOR: Brandon Paulsen
"""

# The smallest a cluster can be before we decide to collapse it into
# another
MIN_CLUSTER_SIZE=6
# If two clusters have atleast this many topic words in common, then
# combine them
CLUSTER_SIM=3

# the topic assignments for documents
# ex: ctx_assignments[i] = j
# means that document with id i was assigned to topic j
ctx_assignments = []
# the raw contexts for the documents
ctxes = []
# a list of sets were a set denotes the topic words for some topic
topics = []
# A string which is the name of the word and the first letter of it's pos
# ex: abandon.v
thing = None
target_word = None
pos = None

"""
FUN6
Converts a senseval2 formatted xml file into the answer key for use by 
the sense cluster scorer script. The answer key is written to
senseclusters_scorer/key
@param f: a string which is a file path to the senseval2 xml file
"""
def makeKey(f):
    global thing
    global target_word
    global pos
    ctx_re = re.compile(r'<instance id="([0-9]*)">.*?<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    mapping = {}
    cur_id = 1
    target, pos = f.split('-', 1)
    pos = re.split('[.-]', pos)[0]
    target_word = target.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        pos = "noun"
        target_word = "xyz"
    thing = target_word + "." + pos[0]
    with open('senseclusters_scorer/key', 'w+') as key_ref:
        for i, (sense_id, sense, ctx) in enumerate(ctx_re.findall(buf)):
            ctxes.append((sense, ctx, sense_id))
            if sense not in mapping:
                mapping[sense] = cur_id
                cur_id += 1
            key_ref.write(thing + " " + thing + "." + str(sense_id) + " " + thing + "." + str(mapping[sense]) + "\n")

"""
Writes the answers generated by hdp-wsi out to a file in the format
that senseclusters_scorer expects. This function should only be called
after readAnswers() has been called. The answer file is written out to
senseclusters_scorer/answers. In addition, also writes out the answers
in a senseval2 formatted xml file, and writes out a definitions file.
Both of these are written to the output directory.
"""
def writeAnswers():
    global pos
    with open('senseclusters_scorer/answers', 'w+') as ans_ref:
        for i, ctx_is in enumerate(ctx_assignments):
            for ctx_i in ctx_is:
                ans_ref.write(thing + " " + thing + "." + str(ctxes[ctx_i][2]) + " " + thing + "." + str(i + 1) + "\n")
    with open('output/' + target_word + '-' + pos + '-assignmnets.xml', 'w+') as f_ref:
        f_ref.write('<corpus lang="english">\n<lexelt item="LEXELT">\n')
        for i, ctx_is in enumerate(ctx_assignments):
            for ctx_i in ctx_is:
                f_ref.write('<instance id="' + str(ctxes[ctx_i][2]) + '">\n')
                f_ref.write('<answer instance="' + str(ctxes[ctx_i][2]) + '" senseid="' + str(i + 1) + '"/>\n')
                f_ref.write('<context>\n')
                f_ref.write(ctxes[ctx_i][1].strip() + '\n')
                f_ref.write('</context>\n</instance>\n')
        f_ref.write('</lexelt>\n</corpus>')
    with open('output/' + target_word + '-' + pos + '.defs', 'w+') as f_ref:
        definition = Definition()
        for i, topic in enumerate(topics):
            if 'noun' in pos or 'verb' in pos:
                f_ref.write(thing + '.' + str(i + 1) + ' definition: ' + definition.generate_Definition(topic, target_word) + '\n')
            else:
                f_ref.write(thing + '.' + str(i + 1) + ' definition: ' + definition.generate_Definition(topic, 'xyz') + '\n')

"""
FUN4
Reads the topics (which are also the senses) generated by hdp-wsi into
memory.
@param f: a string which is the file path to the topic file generated by
          hdp-wsi
"""
def readTopics(f):
    with open(f, 'r') as f_ref:
        for line in f_ref:
            ctx_assignments.append([])
            topics.append(set(line.split(':', 1)[1].strip().split(' ')))
            topics[-1] = filter(lambda x: x != 'NA', topics[-1])
            topics[-1] = set([topic.split('_')[0] for topic in topics[-1]])

"""
FUN5
Reads the topic assigments generated by hdp-wsi into memory.
@param f: a string which is the filepath to the topic assignment file
          generated by hdp-wsi
"""
def readAssignments(f):
    with open(f, 'r') as f_ref:
        for line in f_ref:
            best = float(0)
            best_id = None
            data = line.split(' ')
            ctx_id = int(data[1].split('.')[-1]) - 1
            # COM6
            # A single doc can have multiple assigments to a topic. Take
            # the one with the highest probability
            for assignment in data[2:]:
                t_id, prob = assignment.split('/')
                if float(prob) > best:
                    best = float(prob)
                    best_id = int(t_id.split('.')[1]) - 1
            ctx_assignments[best_id].append(ctx_id)

"""
FUN7
A function that absorbs small clusters (small being defined by MIN_CLUSTER_SIZE) into
a larger cluster which has the most topic words in common with it. We chose not to use
this because it ended up hurting precision, but we are leaving it in because of brandon's
emotional attachment issues.
@return: True if a two clusters were combined, false otherwise
"""
def collapseSmallClusters():
    print "collapse small clusters"
    for i, topic1 in enumerate(topics):
        if len(ctx_assignments[i]) < MIN_CLUSTER_SIZE:
            print "small cluster: " + str(i)
            best = -1
            best_id = -1
            for j, topic2 in enumerate(topics):
                if topic1 is topic2:
                    continue
                elif len(topic1 & topic2) > best:
                    print "new best: " + str(j)
                    best = len(topic1 & topic2)
                    best_id = j
            print "merging topic " + str(i) + " with topic " + str(best_id)
            ctx_assignments[best_id].extend(ctx_assignments[i])
            topics[best_id] |= topics[i]
            del ctx_assignments[i]
            del topics[i]
            return True
    return False


"""
FUN8
Similar to the above function, but collapses together clusters that have at least CLUSTER_SIM
topic words in common. This function should be called until it returns false, or there are
only MIN_CLUSTERS clusters left.
@return: True if a two clusters were combined, false otherwise
"""
def collapseSimilarClusters():
    print "collapse sim clusters"
    for i, topic1 in enumerate(topics):
        best = -1
        best_id = -1
        for j, topic2 in enumerate(topics):
            if topic1 is topic2:
                continue
            elif len(topic1 & topic2) > best and len(topic1 & topic2) >= CLUSTER_SIM:
                    print "new best: " + str(j)
                    best = len(topic1 & topic2)
                    best_id = j
        if best_id == -1: continue
        print "merging topic " + str(i) + " with topic " + str(best_id)
        ctx_assignments[best_id].extend(ctx_assignments[i])
        topics[best_id] |= topics[i]
        del ctx_assignments[i]
        del topics[i]
        return True
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python postProcessing.py <senseval2-xml-file>"
        sys.exit(1)

    # COM4
    # Read the results from hdp-wsi into memory
    readTopics("hdp-wsi/wsi_output/tm_wsi.topics")
    readAssignments("hdp-wsi/wsi_output/tm_wsi")
    # COM5
    # Make the key for senseclusters_scorer
    makeKey(sys.argv[1])

    for i, topic in enumerate(topics):
        print "topic " + str(i) + " words: " + ' '.join(word for word in topic)
        print ' '.join(str(len(topic & topic1)) for topic1 in topics)
        #print "senses: "
        #print ', '.join(ctxes[i][0] for i in ctx_assignments[i])

    while len(topics) > 2:
        if not collapseSmallClusters(): break
    while len(topics) > 2:
        if not collapseSimilarClusters(): break

    for i, topic in enumerate(topics):
        print "topic " + str(i) + " words: " + ' '.join(word for word in topic)
        print ' '.join(str(len(topic & topic1)) for topic1 in topics)
        #print "senses: "
        #print ', '.join(ctxes[i][0] for i in ctx_assignments[i])

    writeAnswers()

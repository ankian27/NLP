import src.config
from collections import defaultdict
import sys
import re

MIN_CLUSTER_SIZE=6

ctx_assignments = []
ctxes = []
topics = []
# A string which is the name of the word and the first letter of it's pos
# ex: abandon.v
thing = None
target_word = None

def makeKey(f):
    global thing
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    mapping = {}
    cur_id = 1
    target, pos = f.split('-', 1)
    target_word = target.split('/')[-1]
    thing = target_word + "." + pos[0]
    with open('senseclusters_scorer/key', 'w+') as key_ref:
        for i, (sense, ctx) in enumerate(ctx_re.findall(buf)):
            ctxes.append((sense, ctx))
            if sense not in mapping:
                mapping[sense] = cur_id
                cur_id += 1
            key_ref.write(thing + " " + thing + "." + str(i + 1) + " " + thing + "." + str(mapping[sense]) + "\n")

def writeAnswers():
    with open('senseclusters_scorer/answers', 'w+') as ans_ref:
        for i, ctx_ids in enumerate(ctx_assignments):
            for ctx_id in ctx_ids:
                ans_ref.write(thing + " " + thing + "." + str(ctx_id + 1) + " " + thing + "." + str(i + 1) + "\n")

def readTopics(f):
    with open(f, 'r') as f_ref:
        for line in f_ref:
            ctx_assignments.append([])
            topics.append(set(line.split(':', 1)[1].strip().split(' ')))

def readAssignments(f):
    with open(f, 'r') as f_ref:
        for line in f_ref:
            best = float(0)
            best_id = None
            data = line.split(' ')
            ctx_id = int(data[1].split('.')[-1]) - 1
            for assignment in data[2:]:
                t_id, prob = assignment.split('/')
                if float(prob) > best:
                    best = float(prob)
                    best_id = int(t_id.split('.')[1]) - 1
            ctx_assignments[best_id].append(ctx_id)

def collapseSmallClusters():
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python postProcessing.py <senseval2-xml-file>"
        sys.exit(1)

    readTopics("hdp-wsi/wsi_output/tm_wsi.topics")
    readAssignments("hdp-wsi/wsi_output/tm_wsi")
    makeKey(sys.argv[1])
    for i, topic1 in enumerate(topics):
        print "topic " + str(i) + " overlap: " + ' '.join(str(len(topic1 & topic2)) for topic2 in topics)
    for i, topic in enumerate(topics):
        print "topic " + str(i) + " words: " + ' '.join(word for word in topic)
        #print "senses: "
        #print ', '.join(ctxes[i][0] for i in ctx_assignments[i])
    # Topic collapsing goes here
    #while collapseSmallClusters(): pass

    writeAnswers()
    # Definition generation goes here

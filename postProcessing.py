from collections import defaultdict
import sys
import re

ctx_assignments = defaultdict(list)
ctxes = []
topics = []

def getCtxes(f):
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'r') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()

def readTopics(f):
    with open(f, 'r') as f_ref:
        for line in f_ref:
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python postProcessing.py <senseval2-xml-file>"
        sys.exit(1)

    readTopics("hdp-wsi/wsi_output/tm_wsi.topics")
    readAssignments("hdp-wsi/wsi_output/tm_wsi")
    ctxes = [ctx for ctx in getCtxes(sys.argv[1])]
    for i, topic1 in enumerate(topics):
        print "topic " + str(i) + " overlap: " + ' '.join(str(len(topic1 & topic2)) for topic2 in topics)
    for i, topic in enumerate(topics):
        print "topic " + str(i) + " words: " + ' '.join(word for word in topic)
        print "senses: "
        print ', '.join(ctxes[i][0] for i in ctx_assignments[i])

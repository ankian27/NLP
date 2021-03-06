import re
"""
@author Brandon Paulsen
"""

"""
A function to extract all the instances from a sense2val file. Not used by the clustering routine.
@param f: a string path to a sense2val file
"""
def get_instances(f):
    instance_re = re.compile(r'<instance id=.*?</instance>', re.MULTILINE | re.DOTALL)
    header_re = re.compile(r'(.*<corpus lang=.*?)<instance',
                re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    
    # pull the header out first
    match = header_re.search(buf)
    if not match:
        # Maybe bad, but header isn't crucial
        print "No header found"
        print f + " might be incorrectly formated"
        header = None
    
    header = match.group(1)

    instances = []
    for instance in instance_re.findall(buf):
        instances.append(instance)

    trailer = "</lexelt>\n</corpus>"
    return header, instances, trailer

"""
Extracts all the contexts from a sense2val file.
@param f: a string path to a sense2value file
@return: an generator which will produce a context everytime it's called
"""
def get_ctxes(f):
    #RE1
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()


"""
Gets the useful information from a sense2val's file name.
@param file_name: a string path to a sense2val file
@return: the target word, the part of speech, and the conflated words if the file name can be recognized 
         as on the name conflate pairs. We need to know the conflated words so we can remove them when we
         are tokenizing a file.
"""
def do_filename(file_name):
    target_word, pos = file_name.split('-', 1)
    pos = pos.split('-', 1)[0]
    target_word = target_word.split('/')[-1]
    conflate_word1 = None
    conflate_word2 = None
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        conflate_word1 = target_word
        conflate_word2 = pos
        target_word = "xyz"
        pos = "noun"
    return target_word, pos, conflate_word1, conflate_word2

"""
Creates a key file from a sense2val file. Outputs the key file in sensecluster_scorer/key.
@param f: a string path to a sense2val file
"""
def make_key(f, target_word, pos):
    thing = target_word + '.' + pos[0].lower()
    ctx_re = re.compile(r'<instance id="[0-9]+">.*?<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>.*?</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    mapping = {}
    cur_id = 1
    with open('senseclusters_scorer/key', 'w+') as key_ref:
        for i, sense in enumerate(ctx_re.findall(buf)):
            if sense not in mapping:
                mapping[sense] = cur_id
                cur_id += 1
            key_ref.write(thing + " " + thing + "." + str(i) + " " + thing + "." + str(mapping[sense]) + "\n")

"""
Creates an answer file from a list of labels. Outputs the answer file in sensecluster_scorer/answers
@param labels: a list of labels like the ones generated from sklean's clustering objects
@param target_word: the target word that was clustered
@param pos: the part of speech as a string
"""
def make_answers(file_name, labels, target_word, pos):
    thing = target_word + '.' + pos[0].lower()
    with open('senseclusters_scorer/answers', 'w+') as ans_ref:
        for i, label in enumerate(labels):
            ans_ref.write(thing + " " + thing + "." + str(i) + " " + thing + "." + str(label + 1) + "\n")
    header, instances, trailer = get_instances(file_name)
    with open('output/' + target_word + '-' + pos + '-assignmnets.xml', 'w+') as f_ref:
        f_ref.write('<corpus lang="english">\n<lexelt item="LEXELT">\n')
        rep_re = re.compile(r'senseid="[^"]+"')
        for i, label in enumerate(labels):
            f_ref.write(rep_re.sub('senseid="' + str(label) + '"', instances[i]) + '\n')
        f_ref.write('</lexelt>\n</corpus>')

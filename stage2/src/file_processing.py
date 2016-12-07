import re

def get_instances(f):
    #RE1
    instance_re = re.compile(r'<instance id=.*?</instance>', re.MULTILINE | re.DOTALL)
    #RE2
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

def get_ctxes(f):
    #RE3
    ctx_re = re.compile(r'<answer.*?senseid="([^"]*)"[^/]*/>.*?<context>(.*?)</context>', re.MULTILINE | re.DOTALL)
    with open(f, 'rb') as f_ref:
        buf = f_ref.read()
    for sense, ctx in ctx_re.findall(buf):
        yield sense, ctx.strip()


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

def make_answers(labels, target_word, pos):
    thing = target_word + '.' + pos[0].lower()
    with open('senseclusters_scorer/answers', 'w+') as ans_ref:
        for i, label in enumerate(labels):
            ans_ref.write(thing + " " + thing + "." + str(i) + " " + thing + "." + str(label + 1) + "\n")



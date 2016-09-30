from src.context import Context
import xmltodict
import optparse
import sys

"""
Disambiguates the given sense2eval xml file, f. The target parameter is only necessary if
the target word isn't marked with <head> tags.
@param f: the file path to the sense2eval xml formatted file
@param target: optional target word
"""
def disam(f, target=None):
    with open(f, 'r') as xml_file:
        xml = xmltodict.parse(xml_file.read())
    for instance in xml['corpus']['lexelt']['instance']:
        #ctx_ref = Context(instance.context)
        print instance['context']
        #print ctx_ref.context
        #print ctx_ref.target
        break


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


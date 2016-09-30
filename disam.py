from src import context
import xmltodict
import optparse
import sys

def disam(f, target=None):
    with open(f, 'r') as xml_file:
        xml = xmltodict.parse(f.read())

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
    disam(f[0], options['target'])


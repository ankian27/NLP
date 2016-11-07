import sys
from src.word2vecModel import Word2VecModel

def main(file_name):
    target_word, pos = file_name.split('-', 1)
    target_word = target_word.split('/')[-1]
    if "noun" not in pos and "verb" not in pos:
        # COM2
        # This is a name conflate pair. The target word is xyz instead of
        # what we find in the file name
        target = "xyz"
        pos = "noun"
    
    return Word2VecModel(file_name, target_word, pos)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage: python writeToHDP.py <senseval2-xml-file>"
        sys.exit(1)
    main(sys.argv[1])

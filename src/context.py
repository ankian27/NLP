import re
import nltk

porter_stemmer = nltk.stem.porter.PorterStemmer()
class Context():

    """
    Constructor
    @param context: the context surrounding the target word. This should include at least
                    25 words to both the left and right of the target word (for a total of
                    50). If the target parameter is not specified, then the target word
                    must be marked with <head> tags
    @param target:  the target of this context. Only needs to be specified if the target
                    word is not marked with <head> tags in the context
    """
    def __init__(self, context, target=None):
        if not target:
            target_re = re.compile(r'<head>(\W*)(\w+)(\W*)</head>')
            self.target = target_re.search(context).group(2)
            self.context = re.sub(r'</?head>', '', context)
        else:
            self.target = target
            self.context = context
        self.tokens = []
        for word in nltk.word_tokenize(self.context):
            if '-' in word:
                for word1 in word.split('-'):
                    self.tokens.append(porter_stemmer.stem(word1.lower()))
            else:
                self.tokens.append(porter_stemmer.stem(word.lower()))
        if target not in self.tokens:
            print ' '.join(self.tokens)
            raise 'could not find target word in tokens'

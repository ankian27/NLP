from gensim.models import Word2Vec
from src.sentenceIter import SentenceIter

si = SentenceIter('sentences')

model = Word2Vec(si, min_count=15, size=200, workers=4)

print "Done training model. Writing it out."
model.save('3_parts_of_wiki_lowercase')

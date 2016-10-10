#This class is used to generate the defitions for each cluster
import nltk 
from nltk.tag import pos_tag, map_tag
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import random

#filename=argv[1]

#topics=set(['grip', 'play', 'ball', 'player', 'tennis' ,'players' ,'racket' ,'table' ,'side' ,'match'])
# topics=set(['agassi' ,'point' ,'american', 'politicians', 'part', 'racket', 'racketeering' ,'called' ,'special'])
#for words in topics:
# target_word='racket'	
#print topics
#text = nltk.word_tokenize("And did you see this? What a comeback. JOE-BUCK: And he hits it in the air to center -- (Crowd-cheering) JOE-BUCK: And we'll you tomorrow night. REBECCA-JARVIS: The St. Louis Cardinals were one <head>strike</head> away from losing the World Series, but they battled back in dramatic fashion to force the first game seven in nearly a decade. We're going to show you how they did it, early this Friday morning, October 28, 2011. I'm going to tell them 100 percent the truth.  McLean is considered a major boon for Maryland football, and not only because his expected signing would <head>strike</head> a blow against Penn State's longstanding recruiting presence in the Washington area. The 6-foot-2, 290-pound McLean has built his reputation on being a prolific prospect both on the field and on social media")



class Definition(object):
	def __init__(self):
		self.cfgRule=defaultdict(list)

	def get_Noun_Verb(self, topics):
		noun=''
		verb=''
		adj=[]
		adv=[]
		# det=['the','a','is','are',]
		# CONJ='and'
		# num = random.randrange(0,2)
		posTagged=nltk.pos_tag(topics)
		simplifiedTags=[(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
		#print simplifiedTags
		for word, tag in simplifiedTags:
			#print word ," ",tag
			if tag=='NOUN':
				# noun.append(word)
				noun += word + '|'
			if tag=='VERB':
				# verb.append(word)
				verb += word+'|'
			if tag=='ADJ':
				adj.append(word)
			if tag=='ADV':
				adv.append(word)
		print noun, verb
		noun=noun[:-1]
		
		verb=verb[:-1]
		return noun, verb

	def cfg_rule(self,left,right):

		rules=right.split('|')
		for rule in rules:
			self.cfgRule[left].append(tuple(rule.split()))

	def gen_def(self, symbol):
		definition = ''
		rule = random.choice(self.cfgRule[symbol])
		for sym in rule:
			if sym in self.cfgRule:
				definition += self.gen_def(sym)
			else:
				definition += sym + ' '
		return definition

	def generate_Definition(self, topics, target):
		if target in topics:
			topics.remove(target)
		sentence = ''
		noun, verb = self.get_Noun_Verb(topics)
		self.cfg_rule('S', 'S1 CONJ S2')
		self.cfg_rule('S1', 'NP VP')
		self.cfg_rule('S2', 'NP VP')
		self.cfg_rule('NP', 'Det N')
		self.cfg_rule('VP', 'V PRO NP')
		self.cfg_rule('CONJ','or | and')
		self.cfg_rule('PRO','with | to')
		self.cfg_rule('Det', 'a | the | is')
		self.cfg_rule('N', noun)
		self.cfg_rule('V', verb)
		sentence += (self.gen_def('S'))	
		print sentence		



#print noun ," ",adj," ",adv," ",verb
#for i in range(3):
#	num = random.randrange(0,3)
#	print noun[num] + ' ' + verb[num] + ' ' + adv[num] + ' ' + adj[num] 



#trigram_measures = nltk.collocations.TrigramAssocMeasures()
#text="And did you see this? What a comeback. JOE-BUCK: And he hits it in the air to center -- (Crowd-cheering) JOE-BUCK: And we'll you tomorrow night. REBECCA-JARVIS: The St. Louis Cardinals were one <head>strike</head> away from losing the World Series, but they battled back in dramatic fashion to force the first game seven in nearly a decade. We're going to show you how they did it, early this Friday morning, October 28, 2011."
#token=nltk.word_tokenize(text)
#tok =RegexpTokenizer(r'[A-Za-z][a-z]+')
#token=tok.tokenize(text)

#print token
#trigram=ngrams(token,3)
#trigram=trigram + '2222'
#lis=list(trigram)
#print lis

#finder=TrigramCollocationFinder.from_words(token)
#ans=finder.nbest(trigram_measures.pmi,10)
#print list(finder)
#for i in range(3):
#	for j in range(3):
        
		#print ans[i][j],
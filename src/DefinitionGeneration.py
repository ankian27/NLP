#This class is used to generate the defitions for each cluster
import nltk 
from nltk.tag import pos_tag, map_tag
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import random

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
 		topics = [topic for topic in topics if target not in topic]
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
		return self.gen_def('S')	

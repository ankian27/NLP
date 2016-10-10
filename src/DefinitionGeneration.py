#This class is used to generate the defitions for each cluster. The idea of definiton generation is that, we can derive the definition of a word by using the context words neighbouring the target word in a given context. The topics are given by the hdp are used to get the topic words. The topic words along with the target_word(the noun/verb/nameconflate pair) is given as input to the program and the output is a sentence generated using those topic words. The sentence gerneated using our approach adheres to the syntactic structure of the enlgish grammar and is more than 10 words. The syntactic structure of the english grammar is represented here in the form of Context Free Grammars(CFG). A CFG is a set of recursive rules(or productions) which are used to generate string patterns. We give the target word as one of the input because if the target word is present in the set of topic words  we want to remove it from the defintion. The execution of the program is as follows:

# Input : Topic words, Target word
# Output: Sentence depicting the meaning of the target word

# Example: shoot woman love look movie director part lot money film
# Output : money love with a movie and a director love with is lot 

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
		""" The function __init__ is a constructor in python which accepts the instance of a class of the object itself as a parameter.
		The constructur is used to initialize the cfgRule(Context Free Grammar rules), nouns, verbs and adjectives for each instance.
		"""
		self.cfgRule=defaultdict(list)
		self.noun = ''
		self.verb = ''
		self.adj  = ''

	def get_Noun_Verb(self, topics):
		"""The function is used to seperate the Nouns, Verbs and Adjectives in the given set of topic words.
		We use the Parts of Speech Tagger from the Natural Language Toolkit to tag the POS for each word in the set of topic words.
		Args:
			param1 (set) : Set of topic words
		Returns:
			Nouns, Verbs and Adjectives seperated from the topic words.
		"""
		self.noun = ''
		self.verb = ''
		self.adj  = ''
		adv=[]

		#Natural Language POS tagger. Returns the default tags
		posTagged=nltk.pos_tag(topics)
		# The default tags are converted to simplified tags. Example: NN->NOUN
		simplifiedTags=[(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
		
		# Seperate Nouns, Verbs and Adjectives by parsing simplifiedTags and assigns to the respective variables.
		for word, tag in simplifiedTags:			
			if tag=='NOUN':
				self.noun += word + '|'
			if tag=='VERB':
				self.verb += word+'|'
			if tag=='ADJ':
				self.adj += word+'|'
			if tag=='ADV':
				adv.append(word)
		
		# Remove the additional '|' character from the strings.
		self.noun=self.noun[:-1]		
		self.verb=self.verb[:-1]
		self.adj=self.adj[:-1]

		return self.noun, self.verb ,self.adj

	def cfg_rule(self,left,right):
		'''The function is used to map the Context Free Grammar production rules for the english grammar to python representation
		Args:
			param1 (string) : Non-terminal String present on the left side of the production
			param2 (string) : Terminal/Non-terminal string present on the right side of the production
		'''

		# Split the string based on '|' character
		rules=right.split('|')
		for rule in rules:
			self.cfgRule[left].append(tuple(rule.split()))

	def gen_def(self, symbol):
		'''The function is used to generate the definition of a sentence recursively using the CFG rules
		Args:
			param1 (string): Start symbol of the CFG rule
		Returns:
			definition: The generated definition of the sentence.
		'''
		definition = ''
		# Randomly select one of the production rule.
		rule = random.choice(self.cfgRule[symbol])
		#Iterate of the symbols of each production rule
		for sym in rule:
			if sym in self.cfgRule:
				definition += self.gen_def(sym)
			else:
				definition += sym + ' '

				noun2=self.noun.split('|')
				verb2=self.verb.split('|')
				
				# Filtering out the already used words. 
				# If a noun has been used, removing it from the list of Noun words.
				noun2 = filter(lambda a: a != sym, noun2) 
				self.noun=''
				# If a verb word has been used, removing it from the list of Verb words.
				verb2 = filter(lambda a: a != sym, verb2)
				self.verb=''

				for words in noun2:
					self.noun += words + '|'
				self.noun=self.noun[:-1]	
				for words in verb2:
					self.verb += words + '|'			
				self.verb=self.verb[:-1]

		return definition

	def generate_Definition(self, topics, target):
		'''This function which is control the flow of program. It makes calls to the functions to produce the CFG rules and to generate the definition of the cluster
		Args:
			param1 (set) : Set of topic words
			param2 (string): The target word for which the definition has to be generated.
		Returns:
			The definition of the target word adhering to the english grammar rules and it is longer than 10 words.
		'''

		# Removes the target word from the set of topic words. As the definition should not contain the word itself.
 		topics = [topic for topic in topics if target not in topic]
 		# Get the seperated Nouns, Verbs, Adjectives
 		self.noun, self.verb ,self.adj= self.get_Noun_Verb(topics)
 		# Represent CFG rules in python
		self.cfg_rule('S', 'S1 CONJ S2')
		self.cfg_rule('S1', 'NP VP')
		self.cfg_rule('S2', 'NP VP')
		self.cfg_rule('NP', 'Det N')
		self.cfg_rule('VP', 'V PRO ADJ NP')
		self.cfg_rule('CONJ','and')
		self.cfg_rule('PRO','with | to')
		self.cfg_rule('Det', 'a | the | is')
		self.cfg_rule('N', self.noun)
		self.cfg_rule('V', self.verb)
		self.cfg_rule('ADJ', self.adj)
		# Generate sentence and return it.
		return self.gen_def('S')	

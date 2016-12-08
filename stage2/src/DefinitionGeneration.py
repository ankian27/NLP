#Author: Ankit Anand Gupta, Sandeep Vuppula
#This class is used to generate the defitions for each cluster. The idea of definiton generation is that, we can derive the definition of a word by using the context words neighbouring the target word in a given context. The topics are given by the hdp are used to get the topic words. The topic words along with the target_word(the noun/verb/nameconflate pair) is given as input to the program and the output is a sentence generated using those topic words. The sentence gerneated using our approach adheres to the syntactic structure of the enlgish grammar and is more than 10 words. The syntactic structure of the english grammar is represented here in the form of Context Free Grammars(CFG). A CFG is a set of recursive rules(or productions) which are used to generate string patterns. We give the target word as one of the input because if the target word is present in the set of topic words  we want to remove it from the defintion. The execution of the program is as follows:

# Input : Topic words, Target word
# Output: Sentence depicting the meaning of the target word

# Example: shoot woman love look movie director part lot money film
# Output : money love with a movie and a director love with is lot 

#The Natural Language Toolkit(NLTK), is an open source toolkit of python modules for natural language processing (NLP) for English language. 
import nltk 
from nltk.tag import pos_tag, map_tag # Function to assign tags to individual tokens and return tagged tokens.
from nltk import word_tokenize # Function to split string of words into individual tokens
from nltk.util import ngrams #Function to return the ngrams generated.
from collections import defaultdict #Creates a default dictionary which gives a default value for non-existent key.
import random #Randomly choose an item from a list of items.
import gensim #Python library for vector space modelling and toolkit modelling
import operator # Has set of functions corresponding to intrinsic operators of python

class Definition(object):
	def __init__(self,model,pos):
		""" 
		#Author: Ankit Anand Gupta
		The function __init__ is a constructor in python which accepts the instance of a class of the object itself as a parameter.
		The constructur is used to initialize the cfgRule(Context Free Grammar rules), nouns, verbs and adjectives for each instance.
		Args:
			param1 (model): The pre-loaded Google model is sent as a parameter for defintion generation instead of reloading it.
			param2 (pos): The parts of speech of the target word.
		"""

		# Create default dictionary
		self.cfgRule=defaultdict(list)
		# Variables to store list of NOUN, VERB and ADJECTIVEs
		self.noun = ''
		self.verb = ''
		self.adj  = ''
		self.model = model
		self.pos = pos

	def get_Noun_Verb(self, topics):
		"""Section I:
		Author: Ankit Anand Gupta
		The function is used to seperate the Nouns, Verbs and Adjectives in the given set of topic words.
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
		
		# Seperate Nouns, Verbs and Adjectives by parsing simplifiedTags and assign to the respective variables.
		# The NOUN words are separated by "|" delimiter
		for word, tag in simplifiedTags:			
			if tag=='NOUN':
				self.noun += word + '|'
			if tag=='VERB':
				self.verb += word+'|'
			if tag=='ADJ':
				self.adj += word+'|'
			if tag=='ADV':
				adv.append(word)
		
		# Remove the additional '|' character from the end of the strings.
		self.noun=self.noun[:-1]		
		self.verb=self.verb[:-1]
		self.adj=self.adj[:-1]

		return self.noun, self.verb ,self.adj

	def cfg_rule(self,left,right):
		'''Section II:
		#Authour: Sandeep Vuppula
		The function is used to map the Context Free Grammar production rules for the english grammar to python representation
		Args:
			param1 (string) : Non-terminal String present on the left side of the production
			param2 (string) : Terminal/Non-terminal string present on the right side of the production
		'''

		# Split the string of Nouns, Verbs, Adjectives appended with "|"
		rules=right.split('|')
		# For each rule of the production, create a tuple and append it to its respective rule in the CFG list.
		for rule in rules:
			self.cfgRule[left].append(tuple(rule.split()))

	def gen_def(self, symbol):
		'''Section III:
		#Author: Sandeep Vuppula
		The function is used to generate the definition of a sentence recursively using the CFG rules
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
			#This condition is true if the sym leads to other nonterminal symbols.
			if sym in self.cfgRule:
				definition += self.gen_def(sym)
			#This is true if the sym leads to terminals. 
			else:
				definition += sym + ' ' # Append the word and the space for the definition.

				# Form a list of nouns and verbs by splitting the string formed above in the function get_Noun_Verb.
				noun2=self.noun.split('|') 
				verb2=self.verb.split('|')
				
				# Filtering out the already used words. 
				# If a noun has been used, removing it from the list of Noun words.
				noun2 = filter(lambda a: a != sym, noun2) 
				self.noun=''
				# If a verb word has been used, removing it from the list of Verb words.
				verb2 = filter(lambda a: a != sym, verb2)
				self.verb=''
				#Repopulating the noun and verb strings with the used word removed.
				for words in noun2:
					self.noun += words + '|'
				self.noun=self.noun[:-1]	
				for words in verb2:
					self.verb += words + '|'			
				self.verb=self.verb[:-1]

		return definition

	def process(self, ctxes, doc_counts):
		'''Section IV:
			#Author: Sandeep Vuppula
			The function takes the ctxes and doc_counts as the parameters. The doc_counts is a dictionary in the format (word,pos)->count. 
			We will sort the doc_counts and then separate nouns, verbs, adjectives into their respective lists. Then we use them to create 
			two parts of the definition. 
				Part 1: Using most_similar word from Word2Vec using topmost nouns(upto 5)
				Part 2: Sentence using the CFG grammar.
			Concatenate Part 1 and Part 2 and return the complete definition.
			Args: 
				param 1 (ctxes):  List of tuples of the form (word, pos, count) for all the contexts in the cluster
				param 2 (doc_counts): Dictionary of the form (word,pos)->count. Count is the number of number of times the 
									  (word,pos) pair appeared in all the contexts in a cluster(even though a pair appears many times in a 
									  single context, it is counted as 1)
			Returns: The generated Definition of the sentence.

		'''
		# Sort the dictionary doc_counts {(word,pos)->count} in descending order.
		doc_counts = sorted(doc_counts.items(), key=operator.itemgetter(1),reverse=True)
		#The noun, verb, adj variables with 'String' suffixes are for generating the definitions using Context Free Grammars
		#The noun, verb, adjectives lists are for forming the definition using the words obtained from most_similar function in Word2Vec.
		nounString = ''
		verbString = ''
		adjString = ''
		nouns = []
		verbs = []
		adjectives = []
		# Separating the (word,pos) into their respective classes of NOUN, VERB, ADJECTIVE and storing just the (word,count)
		for (word,pos), count in doc_counts	:
			if pos == "NOUN":
				nouns.append((word,count))
			if pos == "VERB":
				verbs.append((word,count))
			if pos == "ADJ":
				adjectives.append((word,count))
			else:
				continue
		# Sorting the nouns, verbs, adjectives in reverse order
		nouns.sort(key=lambda tup: tup[1], reverse = True) 
		verbs.sort(key=lambda tup: tup[1], reverse = True)
		adjectives.sort(key=lambda tup: tup[1], reverse = True)
		
		# Considering only top 5 words in list of nouns/verbs/adjectives for forming the sentences using CFG.
		for i,noun in enumerate(nouns):
			if i>4: break
			nounString += noun[0] + '|'
		for i,verb in enumerate(verbs):
			if i>4: break
			verbString += verb[0] + '|'
		for i,adj in enumerate(adjectives):
			if i>4: break
			adjString += adj[0] + '|'

		# Stripping the additional '|'
		nounString =nounString[:-1]		
		verbString=verbString[:-1]
		adjString=adjString[:-1]

		# Different form of sentences are to be formed, based on whether a particular target word is noun/verb.
		# Two parts of the definition are retrieved, concatenated and returned.
		print nouns[:5]
		if "noun" in self.pos:
			ret = 'It is a part of ' + self.createPartOne(nouns, verbs, adjectives) + '. ' + self.generate_Definition(nounString, verbString, adjString)
        	else:
            		ret = 'Something you do with ' + self.createPartOne(nouns, verbs, adjectives) + '. ' + self.generate_Definition(nounString, verbString, adjString)
        	return ret

	def createPartOne(self, nouns, verbs, adjectives):
		'''
		Section V:
		#Author: Ankit Anand Gupta
			This function is the core part for first part of definition generation. We take the top most occurring 5 nouns and then get the 
			most similar word which is related to those words from Word2Vec.
			If there are no nouns at all (possible if cluster formed has very less number of instances) then we return 'unknown'
		Args:
			param 1 (nouns): list of top most nouns
			param 2 (verbs): list of top most verbs
			param 3 (adjectives): set of top most adjectives.

		Return:
			The most similar word retrieved from Word2Vec using the top 5(or less) nouns.
		'''
		# If the list of nouns is empty, return "unknown"
		if not nouns:
			return 'unknown'
		#Get the topmost similar word from Word2Vec.
		word = self.model.most_similar(positive=[noun[0] for noun in nouns[:5]], topn = 1)

		return word[0][0]

	# def generate_Definition(self, topics, target):
	def generate_Definition(self, nounString, verbString, adjString):
		'''Section VI:
		#Author: Ankit Anand Gupta
		This function is the core for second part of the definition generation. It makes calls to the functions to produce the CFG rules and to generate the definition of the cluster
		Args:
			param1 (nounString) : String of nouns, separated by '|'
			param2 (verbStrting): String of verbs, separated by '|'
			param3 (adjStrting)	: String of adjectives, separated by '|'
		Returns:
			The definition formed by applying CFG rules to the noun/verb/adjective Strings.
		'''

		# Removes the target word from the set of topic words. As the definition should not contain the word itself.
        	# topics = filter(lambda topic: target not in topic, topics)
 		# Get the seperated Nouns, Verbs, Adjectives
 		# self.noun, self.verb ,self.adj= self.get_Noun_Verb(topics)
 		# Represent CFG rules in python
 		# S 	 -> S1 CONJ S2
		# S1 	 -> NP VP
		# S2 	 -> NP VP
		# NP 	 -> Det N
		# VP 	 -> V PRO ADJ NP
		# PRO  	 -> with | to
		# Det    -> a | the | is
		# N 	 -> Noun words list
		# V 	 -> Verb words list
		# ADJ    -> Adjective words list
		# CONJ   -> and
		# self.cfg_rule('S', 'S1 CONJ S2')
		self.cfg_rule('S', 'S1')
		self.cfg_rule('S1', 'NP VP')
		# self.cfg_rule('S2', 'NP VP')
		self.cfg_rule('NP', 'Det ADJ N')
		# self.cfg_rule('VP', 'V PRO ADJ NP')
		# self.cfg_rule('VP', 'V PRO NP')
		self.cfg_rule('VP', 'V NP')
		self.cfg_rule('CONJ','and')
		self.cfg_rule('PRO','with | to')
		self.cfg_rule('Det', 'a | the ')
		self.cfg_rule('N', nounString)
		self.cfg_rule('V', verbString)
		self.cfg_rule('ADJ', adjString)
		# Generate sentence and return it.
		return self.gen_def('S')	

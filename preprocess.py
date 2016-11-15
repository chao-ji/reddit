import pandas as pd
import numpy as np
import pickle
import re
import string
import itertools
import nltk

from pandas import DataFrame, Series
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.tag.perceptron import PerceptronTagger

printable = set(string.printable) 

def process_tag(text):
	"""
		Function:	Process urls contained in reddit text.
					Assume input text has been lowercased.
	"""
	sstring = text
	try:
		sstring = sstring.replace("\n", ".").replace("\r", ".")
		# [text](http://...) or [text](https://...)
		sstring = re.sub(r"""\[([^\]]+)\] # [text] parenthesis-captured group \1
                             \(http\S+\)  # (http://...)""", r"\1", sstring, flags=re.X)

		# http://... or https://...
		sstring = re.sub(r"https?://[^\s\"\']+", "", sstring)

		# (...) ... (...) ...
		regex = r"""\(         # left (
                    ([^\(\)]+) # captured text, parenthesis-captured group \1
                    \)         # right )"""

		prnthed_text = re.findall(regex, sstring, flags=re.X)
		sstring = re.sub(regex, "", sstring, flags=re.X)
		for ptext in prnthed_text: 
			sstring = sstring + ". " + ptext

		sstring = sstring.replace("&gt;", "")
		sstring = sstring.replace("&lt;", "")
		sstring = sstring.replace("&amp;", "")
		sstring = sstring.replace("&quot;", "")
		sstring = sstring.replace("&apos;", "")
		sstring = sstring.replace("&cent;", "")
		sstring = sstring.replace("&pound;", "")
		sstring = sstring.replace("&yen;", "")
		sstring = sstring.replace("&euro;", "")
		sstring = sstring.replace("&copy;", "")
		sstring = sstring.replace("&reg;", "")
		sstring = re.sub("[*\[\]\(\)&%\$#@\^]", "", sstring)
	except Exception as inst:
		print inst

	return sstring

def process_pos(text, tagger):
	noun_adjective_list = []

	try:
		tokens = nltk.word_tokenize(text)
		tokens_pos = nltk.tag._pos_tag(tokens, None, tagger)
		for token, pos in tokens_pos:
			if "NN" in pos or "JJ" in pos:
				noun_adjective_list.append(token.encode("ISO-8859-1"))
	except Exception as inst:
		print inst

	return " ".join(noun_adjective_list)

def process_semantic(text):
	sstring = text
	stoplist = nltk.corpus.stopwords.words("english")

	interjections=['aah','ack','agreed','ah','aha','ahem','alas',\
               'all right','amen','argh','as if','aw','ay',\
               'aye','bah','blast','boo hoo','bother','boy',\
               'brr','by golly','bye','cheerio','cheers','chin up',\
               'come on','crikey','curses','dear me','doggone','drat','duh','easy does it','eek','egads','er','exactly',\
               'fair enough','fiddle-dee-dee','fiddlesticks','fie','foo','fooey','  ','gadzooks','gah','gangway','g\'day',\
               'gee','gee whiz','geez','gesundheit','get lost','get outta here','go on','good','good golly','good job',\
               'gosh','gracious','great','grr','gulp','ha','ha-ha','hah','hallelujah','harrumph','haw','hee','here','hey'\
               ,'hmm','ho hum','hoo','hooray','hot dog','how','huh','hum','humbug','hurray','huzza','I say','ick',\
               'im','is it','ixnay','jeez','just kidding','just a sec','just wondering','kapish','la','la-di-dah','lo',\
               'lol','look','look here','long time','lordy','man','meh','mmm','most certainly','my','my my','my word','nah'\
               ,'naw','never','no','no can do','nooo','not','no thanks','no way','nuts','oh','oho','oh-oh','oh no','okay'\
               ,'okey-dokey','om','oof','ooh','oopsey','over','oy','oyez','peace','pff','pew','phew','pish posh','psst','ptui'\
               ,'quite','rah','rats','ready','right','right on','roger','roger that','rumble','say','see ya','shame','shh','shoo'\
               ,'shucks','sigh','sleep tight','snap','sorry','sssh','sup','ta','ta-da','ta ta','take that','tally ho',\
               'tch','thanks','there','there there','time out','tldr','toodles','touche','tsk','tsk-tsk','tut','tut-tut'\
               ,'ugh','uh','uh-oh','um','ur','urgh','very nice','very well','voila','vroom','wah','well','well done',\
               'well, well','what','whatever','whee','when','whoa','whoo','whoopee','whoops','whoopsey','whew','why',\
               'word','wow','wuzzup','ya','yea','yeah','yech','yep','yes','yikes','yippee','yo','yoo-hoo','you bet','you don\'t say',\
                   'you know','yow','yum','yup','yummy','zap','zounds','zowie','zzz']

	exclude = set(string.punctuation.replace("'", ""))
	sstring = "".join(ch for ch in sstring if ch not in exclude)
	try:
		tokens = [token for token in sstring.split(" ") if token not in stoplist and token not in interjections]	
	except Exception as inst:
		print inst
		return text
	return " ".join(tokens)

def get_reddit_tokens(text):
	global printable

	sstring = text
	sstring = filter(lambda x: x in printable, sstring)
	try:
		sstring = sstring.decode("ISO-8859-1").lower()
		sstring = process_tag(sstring)
		tagger = PerceptronTagger()
		sstring = process_pos(sstring, tagger)
		tokens = process_semantic(sstring)
	except Exception as inst:
		print inst
		return []
	return tokens

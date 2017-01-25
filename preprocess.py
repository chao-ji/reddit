import pickle
import re
import string
import itertools
from collections import Iterable

import numpy as np
import pandas as pd
import nltk
from pandas import DataFrame, Series
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.tag.perceptron import PerceptronTagger

printable = set(string.printable) 

def process_tag(text):
    """Process tags (e.g. url and html) contained in reddit text.

    Parameters
    ----------
    text : str (or unicode)
           Raw text to be processed.     

    Returns
    -------
    text : str (or unicode)
           Processed text
    """
    if not isinstance(text, basestring):
        raise TypeError("string format required: got %r" % type(text))

    try:
        text = text.replace("\n", " ").replace("\r", " ")
        # replace `[text](http://...) or [text](https://...)`
        # with `text`
        text = re.sub(r"""\[([^\]]+)\] # [text] parenthesis-captured group \1
                             \(http\S+\)  # (http://...)""", r"\1", text, flags=re.X)

        # replace `http://... or https://...`
        # with ``
        text = re.sub(r"https?://[^\s\"\']+", "", text)

        # replace `(...) ... (...) ...`
        # with ``
        regex = r"""\(         # left (
                    ([^\(\)]+) # captured text, parenthesis-captured group \1
                    \)         # right )"""
        text = re.sub(regex, "", text, flags=re.X)

        text = text.replace("&gt;", "")
        text = text.replace("&lt;", "")
        text = text.replace("&amp;", "")
        text = text.replace("&quot;", "")
        text = text.replace("&apos;", "")
        text = text.replace("&cent;", "")
        text = text.replace("&pound;", "")
        text = text.replace("&yen;", "")
        text = text.replace("&euro;", "")
        text = text.replace("&copy;", "")
        text = text.replace("&reg;", "")
        text = re.sub("[*\[\]\(\)&%\$#@\^]", "", text)
    except Exception as inst:
        print "process_tag: %s" % inst

    return text


def process_semantic(text):
    """Process reddit text semantically. Removes interjections and keep only 
    letters and valid punctuations. 

    Parameters
    ----------
    text : str (or unicode)
           Raw text to be processed.
    Returns
    -------
    text : str (or unicode)
           Processed text
    """
    if not isinstance(text, basestring):
        raise TypeError("string format required: got %r" % type(text))
    
    interjections=['aah','ack','ah','aha','ahem','alas',\
               'all right','amen','argh','aw','ay',\
               'aye','bah','boo hoo',\
               'brr','by golly','bye','cheerio','cheers','chin up',\
               'come on','crikey','dear me','doggone','drat','duh','easy does it','eek','egads','er',\
               'fair enough','fiddle-dee-dee','fiddlesticks','fie','foo','fooey','  ','gadzooks','gah','gangway','g\'day',\
               'gee','gee whiz','geez','gesundheit','get lost','get outta here','go on','good golly',\
               'gosh','grr','gulp','ha','ha-ha','hah','hallelujah','harrumph','haw','hee','hey',\
               'hmm','ho hum','hoo','hooray','huh','hum','humbug','hurray','huzza','I say','ick',\
               'im','ixnay','jeez','just kidding','just a sec','just wondering','kapish','la','la-di-dah','lo',\
               'lol','look here','lordy','meh','mmm','most certainly','my my','my word','nah',\
               'naw','no can do','nooo','no thanks','oh','oho','oh-oh','oh no',\
               'okey-dokey','om','oof','ooh','oopsey','oy','oyez','peace','pff','pew','phew','pish posh','psst','ptui',\
               'rah','right on','roger','roger that','rumble','see ya','shh','shoo',\
               'shucks','sigh','sleep tight','sssh','sup','ta','ta-da','ta ta','tally ho',\
               'tch','there there','time out','tldr','toodles','touche','tsk','tsk-tsk','tut','tut-tut',\
               'ugh','uh', 'uhh', 'uhhh', 'uh-oh','um','ur','urgh','voila','vroom','wah','well done',\
               'well, well','whee','whoa','whoo','whoopee','whoops','whoopsey','whew',\
               'wuzzup','ya','yea','yeah','yech','yep','yes','yikes','yippee','yo','yoo-hoo','you bet','you don\'t say',\
               'you know','yow','yum','yup','yummy','zap','zounds','zowie','zzz']

    valid_char = set(list(string.lowercase) + list(string.digits) + list(" .,;?!'"))
    text = "".join(ch for ch in text if ch in valid_char)
    try:
        tokens = [token for token in text.split(" ") if token not in interjections]    
    except Exception as inst:
        print "process_semantic: %s" % inst

    return " ".join(tokens)


def process_pos(token_list, tagger=PerceptronTagger(), stop_words=stopwords):
    """Process list of tokens in terms of part of speech (POS). Remove any token
    other than noun or adjective 

    Parameters
    ----------
    token_list : list of str, the list of tokens to be processed

    tagger : the POS tagger, default nltk.tag.perceptron.PerceptronTagger()

    stop_words : stopword object, default nltk.corpus.stopwrods

    Returns
    -------
    noun_adjective_list : list of str, the list of tokens tagged as (noun or adjective)
                          from the input
    """
    if not isinstance(token_list, Iterable):
        raise TypeError("input must be iterable: got %r" % type(token_list))

    noun_adjective_list = []
    stop_list = stopwords.words("english")
    try:
        token_list_pos = nltk.tag._pos_tag(token_list, None, tagger)
        noun_adjective_list = [token for token, pos in token_list_pos
                                    if ("NN" in pos or "JJ" in pos) and token not in stop_list]
    except Exception as inst:
        print "process_pos: %s" % inst

    return noun_adjective_list


def get_reddit_tokens(text, nn_adj_only=False):
    """Extract tokens from raw text of reddit post. 
    
    The input will first be tokenized into sentences, and each sentence will be further
    tokenized into words.

    Parameters
    ----------
    text : str (or unicode)
           Raw text to be processed.

    nn_adj_only : bool, whether to keep noun and adjective tokens only, default False

    Returns
    -------
    token_sents : list of list of str (or unicode), list of list of tokens for sentences
    """
    if not isinstance(text, basestring):
        raise TypeError("string format required: got %r" % type(text))

    global printable

    text = filter(lambda x: x in printable, text)
    try:
        text = text.decode("ISO-8859-1").lower()
        text = process_tag(text)
        tokens = process_semantic(text)
        sents = nltk.sent_tokenize(tokens)
        token_sents = map(nltk.word_tokenize, sents)    
        if nn_adj_only:
            token_sents = map(lambda token_list: process_pos(token_list) , token_sents)

    except Exception as inst:
        print "get_reddit_tokens: %s" % inst
        return []
   
    return token_sents

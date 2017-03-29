import re
import sys
import string
import itertools
from collections import Iterable

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag

PRINTABLE = set(string.printable)

STOPLIST = set(stopwords.words("english")) | {"'ve", "'m", "'s", "'re", "'d", "'ll"}

INTERJECTIONS = ['aah','ack','ah','aha','ahem','alas',
                'all right','amen','argh','aw','ay',
                'aye','bah','boo hoo',
                'brr','by golly','bye','cheerio','cheers','chin up',
                'come on','crikey','dear me','doggone','drat','duh','easy does it','eek','egads','er',
                'fair enough','fiddle-dee-dee','fiddlesticks','fie','foo','fooey','  ','gadzooks','gah','gangway','g\'day',
                'gee','gee whiz','geez','gesundheit','get lost','get outta here','go on','good golly',
                'gosh','grr','gulp','ha','ha-ha','hah','hallelujah','harrumph','haw','hee','hey',
                'hmm','ho hum','hoo','hooray','huh','hum','humbug','hurray','huzza','I say','ick',
                'im','ixnay','jeez','just kidding','just a sec','just wondering','kapish','la','la-di-dah','lo',
                'lol','look here','lordy','meh','mmm','most certainly','my my','my word','nah',
                'naw','no can do','nooo','no thanks','oh','oho','oh-oh','oh no',
                'okey-dokey','om','oof','ooh','oopsey','oy','oyez','peace','pff','pew','phew','pish posh','psst','ptui',
                'rah','right on','roger','roger that','rumble','see ya','shh','shoo',
                'shucks','sigh','sleep tight','sssh','sup','ta','ta-da','ta ta','tally ho',
                'tch','there there','time out','tldr','toodles','touche','tsk','tsk-tsk','tut','tut-tut',
                'ugh','uh', 'uhh', 'uhhh', 'uh-oh','um','ur','urgh','voila','vroom','wah','well done',
                'well, well','whee','whoa','whoo','whoopee','whoops','whoopsey','whew',
                'wuzzup','ya','yea','yeah','yech','yep','yes','yikes','yippee','yo','yoo-hoo','you bet','you don\'t say',
                'you know','yow','yum','yup','yummy','zap','zounds','zowie','zzz']

VALID_CHAR = set(list(string.lowercase) + list(string.digits) + list(" .,;?!'-"))

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

        text = text.replace("&nbsp;", "")
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
        text = re.sub("\.{2,}", " ", text)
    except Exception as inst:
        print "process_tag: %s\ninput: %r" % (inst, text)
        sys.exit(1)

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

    try:
        text = "".join(ch for ch in text if ch in VALID_CHAR)
    
        tokens = [token for token in text.split(" ") if token not in INTERJECTIONS]    
    except Exception as inst:
        print "process_semantic: %s\ninput: %r" % (inst, text)
        sys.exit(1)

    text = " ".join(tokens)
    return text


def process_stemming_pos(text,
                        pos_filter=set("VNJ"),
                        stemming=True,
                        stoplist=STOPLIST,
                        sent_tokenize=sent_tokenize,
                        word_tokenize=word_tokenize,
                        lemmantize=WordNetLemmatizer):
    """Perform furthring semantic processing (word stemming, stop-words removal,
    and filtering by part-of-speech)

    Parameters
    ----------
    text : str (or unicode)
           Raw text to be processed.     

    pos_filter : set, optional, default set("VNJ")
                 Set of characters representing part-of-speech to be kept

    stemming : bool, optional, default True
               Whether to perform word stemming

    stoplist : Iterable, optional
               List of stop-words
    
    sent_tokenize : callable, optional, default nltk.sent_tokenize
                    nltk function for tokenizing text into sentences

    word_tokenize : callable, optional, default nltk.word_tokenize
                    nltk function for tokenizing text into words

    lemmantize : callable, optional, default nltk.stem.WordNetLemmatizer
                 nltk function for lemmantizeing (stemming word based on its POS)            

    Returns
    -------
    text : str (or unicode)
           Processed text
    """

    if not isinstance(text, basestring):
        raise TypeError("string format required: got %r" % type(text))
    if not issubclass(type(pos_filter), Iterable) and pos_filter is not None:
        raise TypeError("`pos_filter` must be Iterable: got %r" % type(pos_filter))
    if not isinstance(stemming, bool):
        raise TypeError("`stemming` must be bool: got %r" % type(stemming))
    if not issubclass(type(stoplist), Iterable) and stoplist is not None:
        raise TypeError("`stoplist` must be Iterable: got %r" % type(stoplist))
   
    lemmatize_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ}
    wl = WordNetLemmatizer()

    def stem_word_pos_pair(word, pos):
        v = pos[0].upper()
        if v != "V" and v != "N" and v != "J":
            return word, pos
        else:
            return wl.lemmatize(word, lemmatize_map[v]), pos

    def my_pos_tag(word, pos):
        v = pos[0].upper()
        pos = v if v in pos_filter else '-'
        return (word, pos)

    try:    
        sents_tokens_pos = [pos_tag(word_tokenize(sent))
                            for sent in sent_tokenize(text)]
        tokens_pos = list(itertools.chain(*sents_tokens_pos))
 
        if stoplist is not None:
            tokens_pos = filter(lambda pair: pair[0] not in stoplist, tokens_pos)    
        if stemming:
            tokens_pos = [stem_word_pos_pair(*pair) for pair in tokens_pos]
        if pos_filter is not None:
            tokens_pos = filter(lambda pair: pair[1][0] in pos_filter, tokens_pos)
   
        if len(tokens_pos) == 0:
            return ""

        tokens_pos = filter(lambda pair: re.match(r"[a-z]+\-?[a-z]+$|[0-9]+$", pair[0]) is not None, tokens_pos)
        tokens_pos = map(lambda pair: my_pos_tag(*pair), tokens_pos)
        tokens_pos = map(lambda pair: "/".join(pair), tokens_pos)
        text = " ".join(tokens_pos)

    except Exception as inst:
        print "semantic_processing: %s\ninput: %r" % (inst, text)
        sys.exit(1)

    return text

def get_reddit_tokens(text, njv_only=True):
    """Extract tokens from raw text of reddit post. 
    
    The input will first be tokenized into sentences, and each sentence will be further
    tokenized into words.

    Parameters
    ----------
    text : str (or unicode)
           Raw text to be processed.

    njv_only : bool, whether to keep noun, adjective, verb tokens only, default False

    Returns
    -------
    token_sents : list of list of str (or unicode), list of list of tokens for sentences
    """

    if not isinstance(njv_only, bool):
        raise TypeError("`nvj_only` must be bool: got %r" % type(njv_only))

    text = filter(lambda x: x in PRINTABLE, text)
    text = text.decode("ISO-8859-1").lower()
    text = process_tag(text)
    text = process_semantic(text)
    if njv_only:
        text = process_stemming_pos(text)

    tokens = text.split(" ")
    tokens = filter(lambda x: len(x) > 0, tokens)
    return tokens

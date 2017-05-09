# Reddit -- Understanding the users from polarized online social communities

In human society, people form communities based on shared values, interests, or social practices. This social behavior has also become routine in the online community. For example, subreddits can be viewed as such communities, where people with the same interest can talk to each other under the corresponding subreddit.

The boundaries between online communities can be flexible or rigid. On one hand, people interested in **r/iphone** may be also interested in, say **r/macbook**. On the other hand, some subreddits are so special that users have a sense of "belong-to" or "not-belong-to". A casual example would be, say, **r/PS4** and **r/xboxone**, where people owning different types of consoles would form different online communities. A more serious example would be, for example, **r/MensRights** and **r/Feminism**, which represent two large ideologically opposing and polarized communities on Reddit, so people have a strong sense of belonging and tend to antagonize users from the other community. 

In this data science project, I tried to study **r/MensRights** and **r/Feminism** from different perspective. Some of the questions to be addressed are as follows:

1. Do users have biased choice between **r/MensRights** and **r/Feminism**?
2. What are the most salient linguistic features that distinguish posts in **r/MensRights** from those in **r/Feminism**?
3. How does users' linguistic pattern change over time ?

Below are the highlights of some of the interesting and non-trivial findings:

1. Wordcloud of words over-represented in rMensRights and rFeminism: 
  * rMR specific words: law and legal; sex; family and marriage; curse words
  * rFem specific words: objectification; dressing and body parts; online discussion on reddit; sex; terminologies of ideologies
    
<img src="https://github.com/chao-ji/reddit/blob/master/wordcloud1.png" width="600">
<img src="https://github.com/chao-ji/reddit/blob/master/wordcloud2.png" width="600">
2. Semantic "map" of users from rMensRights and rFeminism. Users are represented as real-valued vectors in multi-dimensional space. Users whose posts are semantically similar are spatially close to each other. Users that are representative of rMR and rFem are colored in orange and cyan
<img src="https://github.com/chao-ji/reddit/blob/master/authorvec1.png" width="600">
3. Posts can be represented as real-valued vectors in multi-dimensional space. We can also find vectors that are most representative of the semantic features of posts in rMR and rFem (semantic poles). The following figure shows that posts semantically close rMR pole received greater scores than posts semantically close to rFem pole.
<img src="https://github.com/chao-ji/reddit/blob/master/score_sempoles.png" width="600">


* [Part 1: Temporal Analysis - Cross-posting Activity](https://github.com/chao-ji/reddit/blob/master/Part%201%20Temporal%20Analysis%20-%20Cross-posting%20Activity%20.ipynb)
* [Part 2: User Analyis - Lifespan](https://github.com/chao-ji/reddit/blob/master/Part%202%20User%20Analyis%20-%20Lifespan%20.ipynb)
* [Part 3: Temporal Analysis - Linguistic Divergence, Unigram BOW ](https://github.com/chao-ji/reddit/blob/master/Part%203%20Temporal%20Analysis%20-%20Linguistic%20Divergence%2C%20Unigram%20BOW%20.ipynb)
* [Part4: Temporal Analysis - Trending Words Over Time](https://github.com/chao-ji/reddit/blob/master/Part%204%20Temporal%20Analysis%20-%20Trending%20Words%20Over%20Time.ipynb)
* [Part 5: Temporal Analysis - Linguistic Divergence, Bigram BOW and Doc2vec Model](https://github.com/chao-ji/reddit/blob/master/Part%205%20Temporal%20Analysis%20-%20Linguistic%20Divergence%2C%20Bigram%20BOW%20and%20Doc2vec%20Model.ipynb)
* [Part 6: Semantic Characterization of Words, Posts, and Authors](https://github.com/chao-ji/reddit/blob/master/Part%206%20Semantic%20Characterization%20of%20Words%2C%20Posts%2C%20and%20Authors.ipynb) 

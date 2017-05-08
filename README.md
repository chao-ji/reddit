# Reddit -- Understanding the users from polarized online social communities

In human society, people form communities based on shared values, interests, or social practices. This social behavior has also become routine in the online community. For example, subreddits can be viewed as such communities, where people with the same interest can talk to each other under the corresponding subreddit.

The boundaries between online communities can be flexible or rigid. On one hand, people interested in **r/iphone** may be also interested in, say **r/macbook**. On the other hand, some subreddits are so special that users have a sense of "belong-to" or "not-belong-to". A casual example would be, say, **r/PS4** and **r/xboxone**, where people owning different types of consoles would form different online communities. A more serious example would be, for example, **r/MensRights** and **r/Feminism**, which represent two large ideologically opposing and polarized communities on Reddit, so people have a strong sense of belonging and tend to antagonize users from the other community. 

In this data science project, I tried to study **r/MensRights** and **r/Feminism** from different perspective. Some of the questions to be addressed are as follows:

1. Do users have biased choice between **r/MensRights** and **r/Feminism**?
2. What are the most salient linguistic features that distinguish posts in **r/MensRights** from those in **r/Feminism**?
3. How does users' linguistic pattern change over time ?

Below are the highlights of some of the interesting and non-trivial findings:

1. rMensRights and rFeminism specific words. 
  A. rMR specific words:
    a. law and legal
    b. sex
    c. family and marriage
    d. curse words
  B. rFem specific words:
    a. objectification
    b. dressing and body parts
    c. online discussion on reddit
    d. sex
    e. terminologies of ideologies
<img src="https://github.com/chao-ji/reddit/blob/master/wordcloud1.png" width="600">
<img src="https://github.com/chao-ji/reddit/blob/master/wordcloud2.png" width="600">
2. Semantic "map" of users from rMensRights and rFeminism
<img src="https://github.com/chao-ji/reddit/blob/master/authorvec1.png" width="600">
3. Posts can be represented as real-valued vectors in multi-dimensional space. We can also find vectors that are most representative of the semantic features of posts in rMR and rFem (semantic poles). The following shows show that posts semantically close rMR pole receive greater scores than posts semantically close to rFem pole.
<img src="https://github.com/chao-ji/reddit/blob/master/score_sempoles.png" width="600">


* [Part 1: Temporal Analysis] (https://github.com/chao-ji/reddit/blob/master/Part%201%20Longitudinal%20Analysis%20-%20Cross-posting%20Activity%20.ipynb)
* [Part 2: Understanding users] (https://github.com/chao-ji/reddit/blob/master/Part%202%20Author%20Lifespan%20.ipynb)
* [Part 3: Linguistic Divergence] (https://github.com/chao-ji/reddit/blob/master/Part%203%20Linguistic%20Divergence%20.ipynb)
* Part4: Shift of Linguistic Pattern
* [Part 5: Topics over Time] (https://github.com/chao-ji/reddit/blob/master/Part%205%20Topics%20Over%20Time%20-%20Overrepresented%20Words.ipynb)
* Part 6: Topic Modeling

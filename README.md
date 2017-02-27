# Reddit -- Studying Association between Linguistic Pattern and the Formation of Polarized Online Communities

In this project, I'm interested in understanding the factors shaping the way people interact with others from a group with opposing interests of values. I mainly used NLP techniques in combination with ad hoc querries and some ETL and exploratory data analysis.

The data was scraped using reddit API (http://files.pushshift.io/reddit/comments/), and is organized as tables where each row corresponds to a reddit post. For each post, four pieces of information were considered:

![author, created_utc, subreddit, body](https://github.com/chao-ji/reddit/blob/master/reddit_data.png)

For the polarized communities I focused on **MensRights** and **Feminism**, as they represent two opposing/conflicting ideologies. The data from these two subreddits will be the focus of this study. 

* [Part 1: Longitudinal Analysis] (https://github.com/chao-ji/reddit/blob/master/Part%201%20Longitudinal%20Analysis%20-%20Cross-posting%20Activity%20.ipynb)
* [Part 2: Author Lifespan] (https://github.com/chao-ji/reddit/blob/master/Part%202%20Author%20Lifespan%20.ipynb)
* [Part 3: Linguistic Divergence] (https://github.com/chao-ji/reddit/blob/master/Part%203%20Linguistic%20Divergence%20.ipynb)
* Part4: Shift of Linguistic Pattern
* [Part 5: Topics over Time] (https://github.com/chao-ji/reddit/blob/master/Part%205%20Topics%20Over%20Time%20-%20Overrepresented%20Words.ipynb)
* Part 6: Topic Modeling

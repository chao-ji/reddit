import datetime 
import pickle
import glob
import os

import pandas as pd
import pymongo
from pymongo import MongoClient
from pandas import DataFrame

client = MongoClient("localhost", 27017)
db = client["reddit_polarization"]

data_path = "/home/jichao/MongoDB/reddit"
bot_file = os.path.join(data_path, "bot_authors_2015_05.csv")
author_bot = pd.read_csv(bot_file)

subreddits = ("MensRights", "Feminism", "Cooking")
for subreddit in subreddits:
    collection = db[subreddit]
    fn_wildcard = os.path.join(data_path, subreddit + "_RC_*.pickle")
    filenames = glob.glob(fn_wildcard)

    for fn in filenames:
        print fn
        df = pickle.load(open(fn))

        df["author"] = df["author"].astype(str)
        df["subreddit"] = df["subreddit"].astype(str)
        df["created_utc"] = df["created_utc"].astype(int)

        # Remove posts from Bots
        df = df.ix[~df["author"].isin(author_bot["author"]), :]
        df["created_utc"] = df["created_utc"].map(lambda x: datetime.datetime.fromtimestamp(x))

        posts = df.T.to_dict().values()

        if len(posts) > 0:
            collection.insert_many(posts)

    collection.create_index([("created_utc", pymongo.ASCENDING)])        

client.close()

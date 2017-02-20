import glob
import pickle
import datetime
from collections import Counter, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from scipy.spatial.distance import cosine
from scipy.stats import entropy


def tokens2counter(row, author_types, radius, srs, fileread):
    """
    Convert list of tokens from specified groups of authors
    to a Counter object
    """
    day = row.name

    days = [day + datetime.timedelta(days=d) for d in [-radius, 0, radius-1]]
    months = ["{0}_{1:02d}".format(d.year, d.month) for d in days]
    months = set(months)
    filenames= [sr + "_" + m + ".pickle" for sr in srs for m in months]

    df = DataFrame()
    for fn in filenames:
        if fn not in fileread:
            try:
                adf = pickle.load(open(fn))
            except IOError as inst:
#                print inst
                continue
            fileread[fn] = adf
        else:
            adf = fileread[fn]
        df = pd.concat([df, adf], axis=0)

    authors = set()
    for t in author_types:
        authors.update(row[t])

    df = df[df["author"].isin(authors)]

    dates = [day + datetime.timedelta(days=d) for d in range(-radius, radius)]
    df = df[df["created_utc"].isin(dates)]

    men = df[df["subreddit"] == "MensRights"]["counter"]
    fem = df[df["subreddit"] == "Feminism"]["counter"]

    men_ct = Counter()
    for ct in men:
        men_ct.update(ct)

    fem_ct = Counter()
    for ct in fem:
        fem_ct.update(ct)

    if len(fileread) > 4:
        fileread.popitem(last=False)
        fileread.popitem(last=False)

    return Series([men_ct, fem_ct], ["from_men", "from_fem"])


def jaccard(row, perc=0.1):
    """Jaccard Similarity"""
    a = row["from_men"]
    b = row["from_fem"]

    len_a = len(a)
    len_b = len(b)

    if int(len_a * perc) >= 1 and int(len_b * perc) >= 1:
        a = a.most_common(int(len_a * perc))
        b = b.most_common(int(len_b * perc))
    else:
        a = a.most_common(len_a)
        b = b.most_common(len_b)
   
    a = set(dict(a).keys())
    b = set(dict(b).keys())

    return len(a & b) / float(len(a | b))

def cosine_sim(row, perc=0.1):
    """cosine similarity"""
    a = row["from_men"]
    b = row["from_fem"]

    len_a = len(a)
    len_b = len(b)

    if int(len_a * perc) >= 1 and int(len_b * perc) >= 1:
        a = a.most_common(int(len_a * perc))
        b = b.most_common(int(len_b * perc))
    else:
        a = a.most_common(int(len_a))
        b = b.most_common(int(len_b))

    a = set(dict(a).keys())
    b = set(dict(b).keys())
    c = a | b
    c = list(c)

    a = np.array(map(lambda x: float(row["from_men"][x]) if x in row["from_men"] else 0., c))
    b = np.array(map(lambda x: float(row["from_fem"][x]) if x in row["from_fem"] else 0., c))
    a = a + 1
    b = b + 1

    r = 1 - cosine(a, b)

    return r

def kldiv(row):
    """KL divergence"""
    a = set(row["from_men"].keys())
    b = set(row["from_fem"].keys())

    c = a | b
    c = list(c)
    a = np.array(map(lambda x: float(row["from_men"][x]) if x in row["from_men"] else 0., c))
    b = np.array(map(lambda x: float(row["from_fem"][x]) if x in row["from_fem"] else 0., c))
    a = a + 1
    b = b + 1
    d = entropy(b, a, base=2.)
    return d

def jsd(row, perc=0.1):
    """Jensen Shannon divergence"""

    a = row["from_men"]
    b = row["from_fem"]

    len_a = len(a)
    len_b = len(b)

    if int(len_a * perc) >= 1 and int(len_b * perc) >= 1:
        a = a.most_common(int(len_a * perc))
        b = b.most_common(int(len_b * perc))
    else:
        a = a.most_common(int(len_a))
        b = b.most_common(int(len_b))

    a = set(dict(a).keys())
    b = set(dict(b).keys())

    c = a | b
    c = list(c)
    a = np.array(map(lambda x: float(row["from_men"][x]) if x in row["from_men"] else 0., c))
    b = np.array(map(lambda x: float(row["from_fem"][x]) if x in row["from_fem"] else 0., c))
    a = a + 1
    b = b + 1
    
    avg = (a + b) / 2.
    return (entropy(a, avg, base=2.) + entropy(b, avg, base=2.)) / 2.



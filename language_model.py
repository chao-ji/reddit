from collections import Iterable
from collections import Counter
import itertools
import numpy as np

class _BaseNGram(object):
    def __init__(self, corpus, special_token=False):
        if not isinstance(corpus, Iterable):
            raise TypeError("corpus must be an Iterable.")
        self._corpus = corpus
        self._special_token = special_token

    def get_mle_count(self, n):
        corpus = self._corpus
        mle_count = Counter()
        for token_iter in corpus:
            if not isinstance(token_iter, Iterable):
                raise TypeError("Each element of corpus must be Iterable.")
            mle_count.update(self.get_ngram_iter(token_iter, n=n,
                special_token=self._special_token))
        mle_count = dict(mle_count)
        if len(mle_count) == 0:
            raise ValueError("corpus must be non-empty.")

        return mle_count

    def get_ngram_iter(self, token_iter, n=1, special_token=False):
        ngram = []
        if special_token:
            if n == 1:
                yield ("<s>",)
            else:
                ngram = ["<s>"] * (n - 1)
        for token in token_iter:
            if not isinstance(token, basestring) or len(token) == 0:
                raise TypeError("token must be non-empty str.")
            ngram.append(token)
            if len(ngram) == n:
                yield tuple(ngram)
                ngram.pop(0)
        if special_token:
            ngram.append("</s>")
            yield tuple(ngram)

class Unigram(_BaseNGram):
    def fit(self, n0=0):
        if not isinstance(n0, int) or n0 < 0:
            raise TypeError("Out of vocabulary size must be an integer >= 0.")
        mle_count = self.get_mle_count(n=1)

        corpus_length = float(sum(mle_count.values()))
        n_mle_count = dict(Counter(mle_count.values()))
        gt_count = dict()
        n_mle_count[0] = n0
        counts = sorted(n_mle_count.keys())
        for r in range(len(counts) - 1):
            gt_count[counts[r]] = counts[r] if n0 == 0 else \
                (counts[r + 1] * n_mle_count[counts[r + 1]]) / \
                float(n_mle_count[counts[r]])
        gt_count[counts[-1]] = counts[-1]

        gt_prob = {k: (v / corpus_length) for k, v in gt_count.iteritems()}

        prob_mass = reduce(lambda x,y: x+y, \
                    map(lambda r: n_mle_count[r] * gt_prob[r], gt_prob.keys()), 0.0)
        gt_prob = {k: v / prob_mass for k, v in gt_prob.iteritems()}

        self._gt_prob = gt_prob
        self._mle_count = mle_count
        self._n_mle_count = n_mle_count

    def predict(self, token_iter, normalize=True):
        if not isinstance(token_iter, Iterable):
            raise TypeError("token_iter must be a Iterable.")
        unigram_iter = self.get_ngram_iter(token_iter, special_token=self._special_token)
        unigram_list = [unigram for unigram in unigram_iter]
        if len(unigram_list) == 0:
            raise ValueError("token_iter must be non-empty.")

        score = reduce(lambda x,y: x+y,
                      map(lambda u: self.logprob(u), unigram_list), 0.0)
        if normalize:
            score = score / float(len(unigram_list))
        return score

    def logprob(self, ngram):
        mle_count = self._mle_count
        gt_prob = self._gt_prob
        if ngram in mle_count:
            return np.log2(gt_prob[mle_count[ngram]])
        else:
            return np.log2(gt_prob[0])
               
class Bigram(_BaseNGram):
    def fit(self, n0=0):
        if not isinstance(n0, int) or n0 < 0:
            raise TypeError("Out of vocabulary size must be an integer >= 0.")

        corpus_iter_bi, corpus_iter_un = itertools.tee(self._corpus)
        self._corpus = corpus_iter_bi
        mle_count = self.get_mle_count(n=2)
        self._corpus = corpus_iter_un
        unigram = Unigram(self._corpus, special_token=self._special_token)
        unigram.fit(n0)

        self._mle_count = mle_count
        self._unigram = unigram

    def predict(self, token_iter, alpha=0.4, normalize=True):
        if not isinstance(token_iter, Iterable):
            raise TypeError("token_iter must be a Iterable.")
        bigram_iter = self.get_ngram_iter(token_iter, n=2, special_token=self._special_token)
        bigram_list = [bigram for bigram in bigram_iter]
        if len(bigram_list) == 0:
            raise ValueError("token_iter must be non-empty.")

        bigram_mle_count = self._mle_count
        unigram_mle_count = self._unigram._mle_count
        unigram = self._unigram

        score = reduce(lambda a,b: a+b, map(lambda x: np.log2(float(bigram_mle_count[x]) / \
                unigram_mle_count[x[:1]]) if x in bigram_mle_count else \
                (np.log2(alpha) + unigram.logprob(x[-1:])), bigram_list), 0.0)

        if normalize:
            score = score / float(len(bigram_list))
        return score

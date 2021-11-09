from typing import List, Dict

from . import base


class LogLDAGibbs(base.LogTopicModel):
    """TODO"""

    def __init__(self, documents, random_seed=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False):
        super().__init__(documents, random_seed=random_seed,
                         stop_words=stop_words,
                         use_nltk_stopwords=use_nltk_stopwords,
                         use_sklearn_stopwords=use_sklearn_stopwords)

        from itertools import chain
        self._vocabulary: List[str] = sorted(set(chain.from_iterable(documents)))
        self._word2id: Dict[str, int] = {w: idx for idx, w
                                         in enumerate(self._vocabulary)}
        self._id2word: Dict[int, str] = {idx: w for idx, w
                                         in enumerate(self._vocabulary)}
        pass

    def vocabulary(self):
        return self._vocabulary

    def word2id(self, word: str):
        return self._word2id[word]

    def id2word(self, corpus_idx: int):
        return self._id2word[corpus_idx]

    def fit(self):
        raise NotImplementedError

    def get_topics(self):
        raise NotImplementedError

    def topic_vector(self, doc: List[str], with_mean=False, with_std=False):
        raise NotImplementedError

    def corpus_topic_matrix(self, corpus=None, with_mean=False, with_std=False):
        raise NotImplementedError

    def topic_terms(self, topic, topn=10):
        raise NotImplementedError

    def all_topic_terms(self, topn=10):
        raise NotImplementedError

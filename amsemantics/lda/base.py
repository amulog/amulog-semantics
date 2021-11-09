from collections import Counter
from abc import ABC, abstractmethod
from typing import Optional, List


class LogTopicModel(ABC):

    def __init__(self, documents: List[List[str]],
                 random_seed: Optional[float] = None,
                 stop_words: Optional[List[str]] = None,
                 use_nltk_stopwords: bool = False,
                 use_sklearn_stopwords: bool = False):
        self._documents = documents
        self._random_seed = random_seed
        self._stop_words = stop_words if stop_words is not None else []
        if use_nltk_stopwords:
            from nltk.corpus import stopwords
            self._stop_words += stopwords.words("english")
        if use_sklearn_stopwords:
            # not recommended for log analysis
            from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
            self._stop_words += list(ENGLISH_STOP_WORDS)

        self._pca = None

    @abstractmethod
    def vocabulary(self):
        raise NotImplementedError

    @abstractmethod
    def word2id(self, word: str):
        raise NotImplementedError

    @abstractmethod
    def id2word(self, corpus_idx: int):
        raise NotImplementedError

    def corpus(self, documents):
        return [self._convert_corpus_elm(doc) for doc in documents]

    def _convert_corpus_elm(self, doc):
        l_wordidx = [self.word2id(w) for w in doc
                     if w not in self._stop_words]
        return list(Counter(l_wordidx).items())

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def get_topics(self):
        raise NotImplementedError

    @abstractmethod
    def topic_vector(self, doc: List[str], with_mean=False, with_std=False):
        raise NotImplementedError

    @abstractmethod
    def corpus_topic_matrix(self, corpus=None, with_mean=False, with_std=False):
        raise NotImplementedError

    @abstractmethod
    def topic_terms(self, topic: int, topn=10):
        raise NotImplementedError

    @abstractmethod
    def all_topic_terms(self, topn=10):
        raise NotImplementedError

    @abstractmethod
    def perplexity(self, corpus=None):
        raise NotImplementedError

    def show_pyldavis(self, mds="pcoa"):
        raise RuntimeError("supporting gensim only")

    def principal_components(self, corpus, n_components=2,
                             with_mean=True, with_std=True):
        if self._pca is None:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=n_components,
                            random_state=self._random_seed)
            base_matrix = self.corpus_topic_matrix(corpus=None,
                                                   with_mean=with_mean,
                                                   with_std=with_std)
            self._pca.fit(base_matrix)

        matrix = self.corpus_topic_matrix(corpus=corpus,
                                          with_mean=with_mean,
                                          with_std=with_std)
        return self._pca.transform(matrix)

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Tuple


class LogTopicModel(ABC):

    def __init__(self,
                 cache_name="ldamodel",
                 cache_dir="/tmp",
                 n_topics: Optional[int] = None,
                 random_seed: Optional[float] = None,
                 stop_words: Optional[List[str]] = None,
                 use_nltk_stopwords: bool = False,
                 use_sklearn_stopwords: bool = False):
        self._cache_name = cache_name
        self._cache_dir = cache_dir
        self._n_topics = n_topics
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
        self._tsne_result = None

    @property
    def n_topics(self):
        return self._n_topics

    def set_n_topics(self, n_topics):
        self._n_topics = n_topics

    def _base_filepath(self):
        name = os.path.join(self._cache_dir, self._cache_name)
        return "{0}_{1}_{2}".format(name, self._n_topics, self._random_seed)

    @abstractmethod
    def cache_exists(self):
        raise NotImplementedError

    @abstractmethod
    def load(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def dump(self):
        raise NotImplementedError

    @abstractmethod
    def vocabulary(self):
        raise NotImplementedError

    @abstractmethod
    def word2id(self, token: str) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def id2word(self, idx: int) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, documents, n_topics=None):
        raise NotImplementedError

    @abstractmethod
    def get_topics(self):
        """Return ndarray of term-topic matrix with shape(n_topics, n_vocabulary)"""
        raise NotImplementedError

    @abstractmethod
    def topic_vector(self, document: List[str], use_zscore=False):
        raise NotImplementedError

    def topic_center(self, documents, use_zscore=False):
        topic_matrix = self.topic_matrix(documents, use_zscore=use_zscore)
        return np.array(topic_matrix).mean(axis=0)

    @abstractmethod
    def topic_matrix(self, documents, use_zscore=False):
        """Return ndarray of gamma matrix (topic weight for each document)
        with shape(n_documents, n_topics)"""
        raise NotImplementedError

    @abstractmethod
    def topic_terms(self, topic: int, topn=10):
        raise NotImplementedError

    @abstractmethod
    def all_topic_terms(self, topn=10):
        raise NotImplementedError

    def topic_center_terms(self, documents, use_zscore=False, topn=10):
        from gensim import matutils
        topic_centerv = self.topic_center(documents, use_zscore=use_zscore)
        topic_centerv = topic_centerv / topic_centerv.sum()
        bestn = matutils.argsort(topic_centerv, topn, reverse=True)
        return [(self.id2word(widx), topic_centerv[widx])
                for widx in bestn]

    def word_top_topics(self, word) -> Iterator[Tuple[int, float]]:
        idx = self.word2id(word)
        a_wordv = self.get_topics()[:, idx]
        for topic_id in np.argsort(a_wordv):
            yield topic_id, a_wordv[topic_id]

    @abstractmethod
    def log_perplexity(self, documents=None):
        raise NotImplementedError

    @abstractmethod
    def perplexity(self, documents=None):
        raise NotImplementedError

    @abstractmethod
    def coherence(self, documents=None):
        raise NotImplementedError

    def show_pyldavis(self, mds="pcoa"):
        raise RuntimeError("supporting gensim only")

    def _pca_model_path(self, n_components, use_zscore):
        return "{0}_pca_{1}_{2}".format(
            self._base_filepath(), n_components, use_zscore
        )

    def exists_pca_model(self, n_components, use_zscore):
        filepath = self._pca_model_path(n_components, use_zscore)
        return os.path.exists(filepath)

    def load_pca_model(self, n_components, use_zscore):
        filepath = self._pca_model_path(n_components, use_zscore)
        from joblib import load
        self._pca = load(filepath)

    def dump_pca_model(self, n_components, use_zscore):
        if self._pca is None:
            self._pca_model(n_components, use_zscore)
        filepath = self._pca_model_path(n_components, use_zscore)
        from joblib import dump
        dump(self._pca, filepath)

    def _pca_model(self, n_components, use_zscore):
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=n_components,
                        random_state=self._random_seed)
        base_matrix = self.topic_matrix(
            documents=None, use_zscore=use_zscore
        )
        self._pca.fit(base_matrix)

    def principal_components(self, documents=None, n_components=2,
                             use_zscore=False):
        if self._pca is None:
            self._pca_model(n_components, use_zscore)

        matrix = self.topic_matrix(
            documents=documents, use_zscore=use_zscore
        )
        return self._pca.transform(matrix)

    def _tsne_path(self, n_components, perplexity, use_zscore):
        return "{0}_tsne_{1}_{2}_{3}".format(
            self._base_filepath(), n_components, perplexity, use_zscore
        )

    def exists_tsne(self, n_components=2, perplexity=30.0, use_zscore=False):
        filepath = self._tsne_path(n_components, perplexity, use_zscore)
        return os.path.exists(filepath)

    def load_tsne(self, n_components=2, perplexity=30.0, use_zscore=False):
        filepath = self._tsne_path(n_components, perplexity, use_zscore)
        from joblib import load
        self._tsne_result = load(filepath)

    def dump_tsne(self, n_components=2, perplexity=30.0,
                  use_zscore=False, n_jobs=None):
        if self._tsne_result is None:
            self._tsne_result = self._calculate_tsne(
                n_components, perplexity, use_zscore, n_jobs=n_jobs
            )
        filepath = self._tsne_path(n_components, perplexity, use_zscore)
        from joblib import dump
        dump(self._tsne_result, filepath)

    def _calculate_tsne(self, n_components, perplexity,
                        use_zscore, n_jobs):
        # unsupervised method (no tsne.transform())
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    random_state=self._random_seed,
                    n_jobs=n_jobs)
        matrix = self.topic_matrix(
            documents=None, use_zscore=use_zscore
        )
        return tsne.fit_transform(matrix)

    def tsne(self, n_components=2, perplexity=30.0,
             use_zscore=False, n_jobs=None):
        if self._tsne_result is None:
            self._tsne_result = self._calculate_tsne(
                n_components, perplexity, use_zscore, n_jobs
            )
        return self._tsne_result

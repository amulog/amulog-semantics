from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class LogTopicModel(ABC):

    def __init__(self, documents: List[List[str]],
                 random_seed: Optional[float] = None,
                 stop_words: Optional[List[str]] = None):
        self._documents = documents
        self._random_seed = random_seed
        self._stop_words = stop_words if stop_words is not None else []

        from itertools import chain
        self._vocabulary: List[str] = sorted(set(chain.from_iterable(documents)))
        self._word2id: Dict[str, int] = {w: idx for idx, w
                                         in enumerate(self._vocabulary)}
        self._id2word: Dict[int, str] = {idx: w for idx, w
                                         in enumerate(self._vocabulary)}

    def vocabulary(self):
        return self._vocabulary

    def word2id(self, word: str):
        return self._word2id[word]

    def id2word(self, corpus_idx: int):
        return self._id2word[corpus_idx]

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def get_topics(self):
        raise NotImplementedError

    @abstractmethod
    def topic_vector(self, doc: List[str]):
        raise NotImplementedError

    @abstractmethod
    def corpus_topic_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def topic_terms(self, topic: int, topn=10):
        raise NotImplementedError

    @abstractmethod
    def all_topic_terms(self, topn=10):
        raise NotImplementedError

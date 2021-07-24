
from . import base


class LogLDAGibbs(base.LogTopicModel):
    """TODO"""

    def __init__(self, documents, random_seed=None,
                 stop_words=None):
        super().__init__(documents, random_seed=random_seed,
                         stop_words=stop_words)
        pass

    def fit(self):
        raise NotImplementedError

    def get_topics(self):
        raise NotImplementedError

    def topic_vector(self, doc):
        raise NotImplementedError

    def corpus_topic_matrix(self):
        raise NotImplementedError

    def topic_terms(self, topic, topn=10):
        raise NotImplementedError

    def all_topic_terms(self, topn=10):
        raise NotImplementedError

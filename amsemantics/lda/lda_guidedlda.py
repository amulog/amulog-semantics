import joblib
import numpy as np
from scipy.stats import zscore

from .lda_gensim import LogLDAgensim


class LogLDAguidedlda(LogLDAgensim):

    refresh = LogLDAgensim.iterations / 10

    def __init__(self, seed_topic_list=None, seed_confidence=0.15,
                 **kwargs):
        super().__init__(**kwargs)
        self._seed_topic_list = seed_topic_list
        self._seed_confidence = seed_confidence

    def load(self):
        model_filepath = self._model_filepath()
        dict_filepath = self._dictionary_filepath()
        input_filepath = self._input_filepath()

        self._ldamodel = joblib.load(model_filepath)
        from gensim.corpora import Dictionary
        self._dictionary = Dictionary.load(dict_filepath)
        self._original_input = joblib.load(input_filepath)

        return self._base_filepath()

    def dump(self):
        model_filepath = self._model_filepath()
        dict_filepath = self._dictionary_filepath()
        input_filepath = self._input_filepath()
        joblib.dump(self._ldamodel, model_filepath)
        self._dictionary.save(dict_filepath)
        joblib.dump(self._original_input, input_filepath)

    def _seed_topics(self):
        if self._seed_topic_list is None:
            return {}

        seed_topics = {}
        for topicid, words in enumerate(self._seed_topic_list):
            for word in words:
                seed_topics[self.word2id(word)] = topicid
        return seed_topics

    @staticmethod
    def _bow_iterator(documents, dictionary):
        for document in documents:
            yield dictionary.doc2bow(document)

    @classmethod
    def _get_term_matrix(cls, documents, dictionary):
        from gensim.matutils import corpus2csc
        bow = cls._bow_iterator(documents, dictionary)
        term_matrix = np.transpose(corpus2csc(bow).astype(np.int64))
        return term_matrix

    def fit(self, documents, n_topics=None):
        if n_topics is None:
            if self._n_topics is None:
                self._n_topics = self.default_n_topics
            n_topics = self._n_topics

        self._dictionary = self._init_dictionary(documents)

        from guidedlda import GuidedLDA
        self._ldamodel = GuidedLDA(
            n_topics=n_topics,
            n_iter=self.iterations,
            random_state=self._random_seed,
            refresh=self.refresh
        )

        self._original_input = self._get_term_matrix(documents, self.dictionary)
        seed_topics = self._seed_topics()

        self._ldamodel.fit(
            self._original_input,
            seed_topics=seed_topics,
            seed_confidence=self._seed_confidence
        )

    def get_topics(self):
        """shape(n_topics, vocabulary_size)"""
        return self.ldamodel.components_

    def topic_vector(self, document, use_zscore=False):
        x_input = self._get_term_matrix([document], self.dictionary)
        v = self.ldamodel.transform(x_input)[0]
        if use_zscore:
            v = np.nan_to_num(zscore(v), nan=float(0))
        return v

    def topic_matrix(self, documents=None, use_zscore=False):
        if documents is None:
            x_input = self._original_input
        else:
            x_input = self._get_term_matrix(documents, self.dictionary)
        matrix = self.ldamodel.transform(x_input)
        if use_zscore:
            matrix = np.nan_to_num(zscore(matrix, axis=1), nan=float(0))
        return matrix

    def _get_topic_terms(self, topicid, topn=10):
        from gensim import matutils
        topicv = self.get_topics()[topicid]
        topicv = topicv / topicv.sum()
        bestn = matutils.argsort(topicv, topn, reverse=True)
        return [(idx, topicv[idx]) for idx in bestn]

    def topic_terms(self, topic, topn=10):
        return [(self.id2word(widx), val)
                for widx, val
                in self._get_topic_terms(topic, topn=topn)]

    def all_topic_terms(self, topn=10):
        return {topic: self.topic_terms(topic, topn=topn)
                for topic in range(self._n_topics)}

    def log_perplexity(self, documents=None):
        raise NotImplementedError

    def perplexity(self, documents=None):
        raise NotImplementedError

    def coherence(self, documents=None):
        raise NotImplementedError

    def show_pyldavis(self, mds="pcoa"):
        raise NotImplementedError("pyldavis does not support guidedlda")

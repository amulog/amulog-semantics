from collections import Counter

from . import base


class LogLDAgensim(base.LogTopicModel):
    """Wrapper of gensim LDA model,
    with some parameters tuned for short sentences such as logs."""

    chunksize = 200
    minimum_probability = 1e-3
    iterations = 500
    default_n_topics = 40

    def __init__(self, documents, n_topics=None, random_seed=None,
                 stop_words=None):
        super().__init__(documents, random_seed=random_seed,
                         stop_words=stop_words)

        self._n_topics = n_topics
        self._corpus = self._init_corpus()

        self._ldamodel = None

    def _init_corpus(self):
        return [self._convert_corpus_elm(doc) for doc in self._documents]

    def _convert_corpus_elm(self, doc):
        l_wordidx = [self.word2id(w) for w in doc
                     if w not in self._stop_words]
        return list(Counter(l_wordidx).items())

    def fit(self, n_topics=None):
        if n_topics is None:
            if self._n_topics is None:
                self._n_topics = self.default_n_topics
            n_topics = self._n_topics

        from gensim.models.ldamodel import LdaModel
        self._ldamodel = LdaModel(
            corpus=self._corpus,
            num_topics=n_topics,
            id2word=self._id2word,
            chunksize=self.chunksize,
            minimum_probability=self.minimum_probability,
            iterations=self.iterations,
            decay=1.0,
            per_word_topics=True,
            random_state=self._random_seed,
        )

    def get_topics(self):
        return self._ldamodel.get_topics()

    def topic_vector(self, doc):
        corpus_elm = self._convert_corpus_elm(doc)
        return self._ldamodel.inference([corpus_elm])[0][0]

    def corpus_topic_matrix(self):
        return self._ldamodel.inference(self._corpus)[0]

    def topic_terms(self, topic, topn=10):
        return [(self._id2word[widx], val)
                for widx, val
                in self._ldamodel.get_topic_terms(topic, topn=topn)]

    def all_topic_terms(self, topn=10):
        return {topic: self.topic_terms(topic, topn=topn)
                for topic in range(self._ldamodel.num_topics)}

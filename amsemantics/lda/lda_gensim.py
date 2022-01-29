import numpy as np
from scipy.stats import zscore

from . import base


class LogLDAgensim(base.LogTopicModel):
    """Wrapper of gensim LDA model,
    with some parameters tuned for short sentences such as logs."""

    chunksize = 200
    minimum_probability = 1e-3
    iterations = 500
    default_n_topics = 40

    def __init__(self, documents, n_topics=None, random_seed=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False):
        super().__init__(documents, random_seed=random_seed,
                         stop_words=stop_words,
                         use_nltk_stopwords=use_nltk_stopwords,
                         use_sklearn_stopwords=use_sklearn_stopwords)

        self._n_topics = n_topics

        self._dictionary = self._init_dictionary(documents)
        self._corpus = self.corpus(documents)

        self._ldamodel = None
        self._vectorizer = None

    def load(self, filepath):
        from gensim.models import LdaModel
        self._ldamodel = LdaModel.load(filepath)

    def dump(self, filepath):
        self._ldamodel.save(filepath)

    def _init_dictionary(self, documents):
        from gensim.corpora import Dictionary
        d = Dictionary(documents)

        bad_ids = []
        for sword in self._stop_words:
            if sword in d.token2id:
                bad_ids.append(d.token2id[sword])
        d.filter_tokens(bad_ids=bad_ids)

        return d

    def vocabulary(self):
        return self._dictionary.token2id.keys()

    def word2id(self, word: str):
        if word in self._dictionary.token2id:
            return self._dictionary.token2id[word]
        else:
            return None

    def id2word(self, corpus_idx: int):
        if corpus_idx in self._dictionary.id2token:
            return self._dictionary.id2token[corpus_idx]
        else:
            return None

    def fit(self, n_topics=None):
        if n_topics is None:
            if self._n_topics is None:
                self._n_topics = self.default_n_topics
            n_topics = self._n_topics

        from gensim.models.ldamodel import LdaModel
        self._ldamodel = LdaModel(
            corpus=self._corpus,
            num_topics=n_topics,
            id2word=self._dictionary,
            chunksize=self.chunksize,
            minimum_probability=self.minimum_probability,
            iterations=self.iterations,
            decay=1.0,
            per_word_topics=True,
            random_state=self._random_seed,
        )

    def get_topics(self):
        return self._ldamodel.get_topics()

    def topic_vector(self, doc, use_zscore=False):
        corpus_elm = self._convert_corpus_elm(doc)
        v = self._ldamodel.inference([corpus_elm])[0][0]
        # v = scale(v, with_mean=with_mean, with_std=with_std)
        if use_zscore:
            v = np.nan_to_num(zscore(v), nan=float(0))
        return v

    def corpus_topic_matrix(self, corpus=None, use_zscore=False):
        if corpus is None:
            corpus = self._corpus

        matrix = self._ldamodel.inference(corpus)[0]
        # matrix = scale(matrix, axis=1, with_mean=with_mean, with_std=with_std)
        if use_zscore:
            matrix = np.nan_to_num(zscore(matrix, axis=1), nan=float(0))
            if np.isnan(matrix).any().sum():
                import pdb; pdb.set_trace()
        return matrix

    def topic_terms(self, topic, topn=10):
        return [(self.id2word(widx), val)
                for widx, val
                in self._ldamodel.get_topic_terms(topic, topn=topn)]

    def all_topic_terms(self, topn=10):
        return {topic: self.topic_terms(topic, topn=topn)
                for topic in range(self._ldamodel.num_topics)}

    def log_perplexity(self, corpus=None):
        if corpus is None:
            corpus = self._corpus
        return self._ldamodel.log_perplexity(corpus)

    def perplexity(self, corpus=None):
        if corpus is None:
            corpus = self._corpus
        # log_perplexity returns bound where p = e^(-bound)
        # (gensim document says 2^(-bound), but not in gensim source code)
        return np.exp(-self._ldamodel.log_perplexity(corpus))

    def coherence(self, corpus=None):
        if corpus is None:
            corpus = self._corpus
        from gensim.models import CoherenceModel
        cm = CoherenceModel(self._ldamodel, corpus=corpus,
                            dictionary=self._dictionary, coherence='u_mass')
        return cm.get_coherence()

    def show_pyldavis(self, mds="pcoa"):
        import pyLDAvis.gensim_models
        return pyLDAvis.gensim_models.prepare(
            self._ldamodel,
            self._corpus,
            self._dictionary,
            mds=mds
        )

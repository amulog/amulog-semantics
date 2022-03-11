import numpy as np
from collections import Counter
from scipy.stats import zscore

from . import base


class LogLDAgensim(base.LogTopicModel):
    """Wrapper of gensim LDA model,
    with some parameters tuned for short sentences such as logs."""

    chunksize = 200
    minimum_probability = 1e-3
    iterations = 500
    default_n_topics = 40

    def __init__(self,
                 cache_name="ldamodel",
                 cache_dir="/tmp",
                 n_topics=None, random_seed=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False):
        super().__init__(
            cache_name=cache_name,
            cache_dir=cache_dir,
            n_topics=n_topics,
            random_seed=random_seed,
            stop_words=stop_words,
            use_nltk_stopwords=use_nltk_stopwords,
            use_sklearn_stopwords=use_sklearn_stopwords
        )

        self._ldamodel = None
        self._dictionary = None
        self._original_input = None

    @property
    def ldamodel(self):
        if self._ldamodel is None:
            raise ValueError("LDA model has not been generated or loaded")
        else:
            return self._ldamodel

    @property
    def dictionary(self):
        if self._dictionary is None:
            raise ValueError("LDA dictionary has not been generated or loaded")
        else:
            return self._dictionary

    @property
    def original_input(self):
        if self._original_input is None:
            raise ValueError("corpus has not been generated or loaded")
        else:
            return self._original_input

    def _model_filepath(self):
        return self._base_filepath() + "_model"

    def _dictionary_filepath(self):
        return self._base_filepath() + "_dict"

    def _input_filepath(self):
        return self._base_filepath() + "_corpus"

    def cache_exists(self):
        import os.path
        model_filepath = self._model_filepath()
        dict_filepath = self._dictionary_filepath()
        corpus_filepath = self._input_filepath()
        exist_model = os.path.exists(model_filepath)
        exist_dict = os.path.exists(dict_filepath)
        exist_corpus = os.path.exists(corpus_filepath)
        return exist_model and exist_dict and exist_corpus

    def load(self):
        model_filepath = self._model_filepath()
        dict_filepath = self._dictionary_filepath()
        input_filepath = self._input_filepath()

        from gensim.models import LdaModel
        from gensim.corpora import Dictionary, MmCorpus
        self._ldamodel = LdaModel.load(model_filepath)
        self._dictionary = Dictionary.load(dict_filepath)
        self._original_input = MmCorpus(input_filepath)

        return self._base_filepath()

    def dump(self):
        model_filepath = self._model_filepath()
        dict_filepath = self._dictionary_filepath()
        corpus_filepath = self._input_filepath()
        self._ldamodel.save(model_filepath)
        self._dictionary.save(dict_filepath)
        from gensim.corpora import MmCorpus
        MmCorpus.serialize(corpus_filepath, self._original_input)

    def _init_dictionary(self, documents):
        if documents is None:
            return None

        from gensim.corpora import Dictionary
        d = Dictionary(documents)

        bad_ids = []
        for sword in self._stop_words:
            if sword in d.token2id:
                bad_ids.append(d.token2id[sword])
        d.filter_tokens(bad_ids=bad_ids)

        return d

    def vocabulary(self):
        return self.dictionary.token2id.keys()

    def word2id(self, token: str):
        if token in self.dictionary.token2id:
            return self.dictionary.token2id[token]
        else:
            return None

    def id2word(self, idx: int):
        if idx in self.dictionary:
            return self.dictionary[idx]
        else:
            return None

    def corpus(self, documents):
        return [self._convert_corpus_elm(doc) for doc in documents]

    def _convert_corpus_elm(self, doc):
        vocabulary = self.vocabulary()
        l_wordidx = [self.word2id(w) for w in doc
                     if w in vocabulary and w not in self._stop_words]
        return list(Counter(l_wordidx).items())

    def fit(self, documents, n_topics=None):
        if n_topics is None:
            if self._n_topics is None:
                self._n_topics = self.default_n_topics
            n_topics = self._n_topics

        self._dictionary = self._init_dictionary(documents)
        self._original_input = self.corpus(documents)

        from gensim.models.ldamodel import LdaModel
        self._ldamodel = LdaModel(
            corpus=self._original_input,
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

    def topic_vector(self, document, use_zscore=False):
        corpus_elm = self._convert_corpus_elm(document)
        v = self._ldamodel.inference([corpus_elm])[0][0]
        if use_zscore:
            v = np.nan_to_num(zscore(v), nan=float(0))
        return v

    def topic_matrix(self, documents=None, use_zscore=False):
        if documents is None:
            corpus = self.original_input
        else:
            corpus = self.corpus(documents)
        matrix = self._ldamodel.inference(corpus)[0]
        if use_zscore:
            matrix = np.nan_to_num(zscore(matrix, axis=1), nan=float(0))
        return matrix

    def topic_terms(self, topic, topn=10):
        return [(self.id2word(widx), val)
                for widx, val
                in self.ldamodel.get_topic_terms(topic, topn=topn)]

    def all_topic_terms(self, topn=10):
        return {topic: self.topic_terms(topic, topn=topn)
                for topic in range(self.ldamodel.num_topics)}

    def log_perplexity(self, documents=None):
        if documents is None:
            corpus = self.original_input
        else:
            corpus = self.corpus(documents)
        return self.ldamodel.log_perplexity(corpus)

    def perplexity(self, documents=None):
        if documents is None:
            corpus = self.original_input
        else:
            corpus = self.corpus(documents)
        # log_perplexity returns bound where p = e^(-bound)
        # (gensim document says it is 2^(-bound), but not in gensim source code)
        return np.exp(-self.ldamodel.log_perplexity(corpus))

    def coherence(self, documents=None):
        if documents is None:
            corpus = self.original_input
        else:
            corpus = self.corpus(documents)
        from gensim.models import CoherenceModel
        cm = CoherenceModel(self.ldamodel, corpus=corpus,
                            dictionary=self.dictionary, coherence='u_mass')
        return cm.get_coherence()

    def show_pyldavis(self, mds="pcoa"):
        import pyLDAvis.gensim_models
        return pyLDAvis.gensim_models.prepare(
            self.ldamodel,
            self.original_input,
            self.dictionary,
            mds=mds
        )

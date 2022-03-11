import pandas as pd
from typing import Optional, List


class SemanticClassifier:

    def __init__(self,
                 training_sources: Optional[List[str]] = None,
                 input_sources: Optional[List[str]] = None,
                 normalizer=None,
                 lda_library="gensim",
                 use_cache=False,
                 cache_dir="/tmp",
                 rfc_document_unit="rfc",
                 ltjunos_filepath=None,
                 use_template_replacer=False,
                 random_seed=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False,
                 lda_n_topics=None,
                 lda_use_zscore=True,
                 lda_cachename="ldamodel",
                 guidedlda_seed_topic_list=None,
                 guidedlda_seed_confidence=0.15,
                 cluster_method="DBSCAN",
                 dbscan_eps=None,
                 dbscan_cluster_size_min=5,
                 dbscan_tuning_metrics="cluster",
                 rdbscan_cluster_size_max=20,
                 ):
        self._lda_library = lda_library
        self._lognorm = normalizer
        self._use_cache = use_cache
        self._cache_dir = cache_dir
        self._rfc_document_unit = rfc_document_unit
        self._ltjunos_filepath = ltjunos_filepath
        self._use_template_replacer = use_template_replacer
        self._stop_words = stop_words
        self._use_nltk_stopwords = use_nltk_stopwords
        self._use_sklearn_stopwords = use_sklearn_stopwords
        self._random_seed = random_seed
        self._lda_n_topics = lda_n_topics
        self._lda_use_zscore = lda_use_zscore
        self._lda_cachename = lda_cachename
        self._guidedlda_seed_topic_list = guidedlda_seed_topic_list
        self._guidedlda_seed_confidence = guidedlda_seed_confidence
        self._cluster_method = cluster_method
        self._dbscan_eps = dbscan_eps
        self._dbscan_cluster_size_min = dbscan_cluster_size_min
        self._dbscan_tuning_metrics = dbscan_tuning_metrics
        self._rdbscan_cluster_size_max = rdbscan_cluster_size_max

        assert isinstance(self._use_cache, bool)
        assert isinstance(self._use_nltk_stopwords, bool)
        assert isinstance(self._use_sklearn_stopwords, bool)
        assert isinstance(self._lda_use_zscore, bool)

        if training_sources is None:
            training_sources = []
        self._training_sources = training_sources
        if input_sources is None:
            input_sources = []
        self._input_sources = input_sources

        self._loglda = self._init_loglda()
        self._tcls = self._init_topic_clustering()

        self._given_training_docs = []
        self._given_training_annotations = []
        self._given_input_docs = []
        self._given_input_annotations = []

        # lazy evaluation
        self._training_docs = None
        self._training_annotations = None
        self._input_docs = None
        self._input_annotations = None

        # used after set_tuning_rules
        self._tuning_rules = None

    def get_input_documents(self, verbose=False):
        if self._input_docs is None:
            if len(self._input_sources) == 0:
                self._input_docs = self._given_input_docs[:]
                self._input_annotations = self._given_input_annotations
            else:
                self._load_documents(verbose=verbose)
        assert self._input_docs is not None
        return self._input_docs

    def get_training_documents(self, verbose=False):
        if self._training_docs is None:
            if self._input_docs is None:
                self._load_documents(verbose=verbose)
            else:
                self._load_training_documents(verbose=verbose)
        assert self._training_docs is not None
        return self._training_docs

    def _document_source(self, source_name):
        if source_name == "rfc":
            from .source import RFCSource
            kwargs = {"document_unit": self._rfc_document_unit,
                      "use_cache": self._use_cache,
                      "cache_dir": self._cache_dir,
                      "normalizer": self._lognorm}
            return RFCSource(**kwargs)
        elif source_name == "ltjunos":
            from .source import LTJunosSource
            return LTJunosSource(
                self._ltjunos_filepath, self._lognorm,
                use_replacer=self._use_template_replacer
            )
        else:
            raise ValueError

    def _load_training_documents(self, verbose=False):
        self._training_docs = self._given_training_docs[:]
        self._training_annotations = self._given_training_annotations[:]
        if verbose:
            print("load document sources: {0}".format(self._training_sources))

        for source_name in self._training_sources:
            source = self._document_source(source_name)
            for document, description in source.load(verbose=verbose):
                self._training_docs.append(document)
                self._training_annotations.append(description)

    def _load_documents(self, verbose=False):
        self._training_docs = self._given_training_docs[:]
        self._training_annotations = self._given_training_annotations[:]
        self._input_docs = self._given_input_docs[:]
        self._input_annotations = self._given_input_annotations

        document_sources = set(self._training_sources + self._input_sources)
        if verbose:
            print("load document sources: {0}".format(document_sources))

        for source_name in document_sources:
            source = self._document_source(source_name)
            docs = []
            annotations = []
            for document, description in source.load(verbose=verbose):
                docs.append(document)
                annotations.append(description)
            if source_name in self._training_sources:
                self._training_docs += docs
                self._training_annotations += annotations
            if source_name in self._input_sources:
                self._input_docs += docs
                self._input_annotations += annotations

    def add_input_documents(self, documents, annotations=None):
        assert self._input_docs is None, SyntaxWarning
        self._given_input_docs += documents
        if annotations is None:
            self._given_input_annotations += [None] * len(documents)
        else:
            self._given_input_annotations += annotations

    def add_training_documents(self, documents, annotations=None):
        assert self._training_docs is None, SyntaxWarning
        self._given_training_docs += documents
        if annotations is None:
            self._given_training_annotations += [None] * len(documents)
        else:
            self._given_training_annotations += annotations

    def _init_loglda(self):
        if self._lda_library == "gensim":
            from amsemantics.lda import LogLDAgensim
            return LogLDAgensim(
                cache_name=self._lda_cachename,
                cache_dir=self._cache_dir,
                n_topics=self._lda_n_topics,
                random_seed=self._random_seed,
                stop_words=self._stop_words,
                use_nltk_stopwords=self._use_nltk_stopwords,
                use_sklearn_stopwords=self._use_sklearn_stopwords,
            )
        elif self._lda_library == "guidedlda":
            from amsemantics.lda import LogLDAguidedlda
            return LogLDAguidedlda(
                seed_topic_list=self._guidedlda_seed_topic_list,
                seed_confidence=self._guidedlda_seed_confidence,
                cache_name=self._lda_cachename,
                cache_dir=self._cache_dir,
                n_topics=self._lda_n_topics,
                random_seed=self._random_seed,
                stop_words=self._stop_words,
                use_nltk_stopwords=self._use_nltk_stopwords,
                use_sklearn_stopwords=self._use_sklearn_stopwords,
            )
        elif self._lda_library == "gibbs":
            raise NotImplementedError
        else:
            raise ValueError

    def exists_lda_cache(self):
        return self._loglda.cache_exists()

    def load_lda(self):
        assert self.exists_lda_cache()
        self._loglda.load()

    def dump_lda(self):
        self._loglda.dump()

    def make_lda(self, verbose=False):
        training_docs = self.get_training_documents(verbose=verbose)
        self._loglda.fit(training_docs)

    def fit_or_load_loglda(self, verbose=False):
        if self._use_cache:
            if self.exists_lda_cache():
                self.load_lda()
                if verbose:
                    print("load LDA cache ({0})".format(
                        self._lda_cachename, self._cache_dir
                    ))
            else:
                self.make_lda(verbose=verbose)
                self.dump_lda()
                if verbose:
                    print("dump LDA cache ({0} in {1})".format(
                        self._lda_cachename, self._cache_dir
                    ))
        else:
            self.make_lda(verbose=verbose)

    def set_n_topics(self, n_topics):
        """n_topics=None will initialize n_topics parameter (for param tuning)"""
        self._lda_n_topics = n_topics
        self._loglda.set_n_topics(n_topics)

    def set_tuning_rules(self, *args, **kwargs):
        self._tuning_rules = (args, kwargs)

    def _init_topic_clustering(self):
        if self._cluster_method == "DBSCAN":
            from amsemantics import TopicClusteringDBSCAN
            return TopicClusteringDBSCAN(
                dbscan_eps=self._dbscan_eps,
                cluster_size_min=self._dbscan_cluster_size_min,
                redistribute=True,
            )
        elif self._cluster_method == "RecursiveDBSCAN":
            from amsemantics import TopicClusteringRecursiveDBSCAN
            return TopicClusteringRecursiveDBSCAN(
                dbscan_eps=self._dbscan_eps,
                recursive_cluster_size=self._rdbscan_cluster_size_max,
            )
        else:
            raise NotImplementedError

    def _make_clustering_with_autotuning(self, verbose=False):
        from amsemantics import ParameterTuning
        pt = ParameterTuning(
            tuning_metrics=self._dbscan_tuning_metrics,
            cache_dir=self._cache_dir,
            lda_use_zscore=self._lda_use_zscore,
            lda_use_cache=self._use_cache,
            lda_cachename=self._lda_cachename,
        )
        if self._tuning_rules is not None:
            args, kwargs = self._tuning_rules
            pt.set_tuning_rules(*args, **kwargs)

        input_docs = self.get_input_documents(verbose=verbose)

        if self._lda_n_topics is None:
            training_docs = self.get_training_documents(verbose=verbose)
            if self._tcls.need_tuning():
                # tune params for both LDA and clustering
                n_topics, tcls_params = pt.tune_all(
                    self._loglda, self._tcls,
                    training_docs, input_docs, verbose=verbose
                )
                # generate loglda model with tuned params
                self.set_n_topics(n_topics)
                self._tcls.set_param(tcls_params)
                self.fit_or_load_loglda(verbose=verbose)
            else:
                # tune params for LDA
                n_topics = pt.tune_lda(
                    self._loglda, self._tcls,
                    training_docs, input_docs, verbose=verbose
                )
                # generate loglda model with tuned params
                self.set_n_topics(n_topics)
                self.fit_or_load_loglda(verbose=verbose)
        else:
            # generate loglda model with given params
            self.fit_or_load_loglda(verbose=verbose)
            if self._tcls.need_tuning():
                # tune params for clustering
                tcls_params = pt.tune_clustering(
                    self._loglda, self._tcls, input_docs, verbose=verbose
                )
                self._tcls.set_param(tcls_params)

        # generate clusters with tuned (or given) params
        input_corpus = self._loglda.corpus(input_docs)
        topic_matrix = self._loglda.topic_matrix(
            input_corpus, use_zscore=self._lda_use_zscore
        )
        self._tcls.fit(topic_matrix)

    def make(self, verbose=False):
        self._make_clustering_with_autotuning(verbose=verbose)
        return self._tcls.cluster_labels, self._input_annotations

    def clusters(self):
        if self._tcls.clusters is None:
            raise ValueError
        else:
            return self._tcls.clusters

    def cluster_labels(self):
        if self._tcls.cluster_labels is None:
            raise ValueError
        else:
            return self._tcls.cluster_labels

    def input_annotations(self):
        if self._input_annotations is None:
            self.get_input_documents()
        return self._input_annotations

    def training_annotations(self):
        if self._training_annotations is None:
            self.get_training_documents()
        return self._training_annotations

    def lda_search_params(self, verbose=False):
        original_lda_n_topics = self._lda_n_topics
        training_docs = self.get_training_documents(verbose=verbose)

        from amsemantics import ParameterTuning
        results = []
        for n_topics in ParameterTuning.get_lda_param_candidates(training_docs):
            self._lda_n_topics = n_topics
            self._loglda.set_n_topics(n_topics)
            self.fit_or_load_loglda(verbose=verbose)

            lp = self._loglda.log_perplexity()
            coherence = self._loglda.coherence()
            results.append((n_topics, lp, coherence))

        self._lda_n_topics = original_lda_n_topics
        self._loglda.set_n_topics(original_lda_n_topics)

        return pd.DataFrame(
            results,
            columns=["n_topics", "log_perplexity", "coherence"]
        )

    def survey_documents(self, documents):
        return {
            "topic_center": self._loglda.topic_center(
                documents, use_zscore=False
            ),
        }

    def surver_word(self, word):
        return {
            "top_topics": self._loglda.word_top_topics(word),
        }

    def get_inspection(self, topn=10):
        d = {
            "topic_terms": self._loglda.all_topic_terms(topn),
        }
        if self._tcls is not None:
            d["cluster_terms"] = self._tcls.all_cluster_terms(self._loglda, topn)

        return d

    def get_visual_data(self, documents=None, dim=2, method="pca",
                        n_jobs=None, verbose=False):
        if method == "pca":
            if documents is None:
                corpus = None
            else:
                corpus = self._loglda.corpus(documents)
            kwargs = {"n_components": dim,
                      "use_zscore": self._lda_use_zscore}
            if self._use_cache:
                if self._loglda.exists_pca_model(**kwargs):
                    if verbose:
                        print("load pca model cache")
                    self._loglda.load_pca_model(**kwargs)
                else:
                    self._loglda.dump_pca_model(**kwargs)
                    if verbose:
                        print("dump pca model cache")
            return self._loglda.principal_components(corpus, **kwargs)
        elif method == "tsne":
            if documents is not None:
                raise SyntaxWarning("tsne does not accept document input"
                                    "because tsne is unsupervised")
            kwargs = {"n_components": dim,
                      "use_zscore": self._lda_use_zscore}
            if self._use_cache:
                if self._loglda.exists_tsne(**kwargs):
                    if verbose:
                        print("load tsne result cache")
                    self._loglda.load_tsne(**kwargs)
                else:
                    self._loglda.dump_tsne(n_jobs=n_jobs, **kwargs)
                    if verbose:
                        print("dump tsne result cache")
            return self._loglda.tsne(n_jobs=n_jobs, **kwargs)

    def show_lda_visual(self, domains=None, keywords=None,
                        colormap=None, title="plot",
                        method="pca", n_jobs=None, verbose=False):
        """Show interactive visual mapping of documents with bokeh.
        This method only consider LDA topic space (no clustering results).
        This method is assumed used in jupyter notebook or lab.

        Returns:
            bokeh.Figure
        """
        import numpy as np
        from bokeh.plotting import figure, ColumnDataSource

        visual_matrix = self.get_visual_data(
            method=method, n_jobs=n_jobs, verbose=verbose
        )
        annotations = np.array(self.training_annotations())
        a_domain = annotations[:, 0]

        if colormap is None:
            from bokeh.palettes import Category10
            domain_names = set(a_domain)
            domain_colormap = {domain_name: Category10[max(3, len(domain_names))][i]
                               for i, domain_name in enumerate(domain_names)}
        else:
            domain_colormap = colormap

        if domains is None:
            c0 = visual_matrix[:, 0]
            c1 = visual_matrix[:, 1]
            a_description = annotations[:, 1]
            a_colors = np.array([domain_colormap[domain_name]
                                 for domain_name in a_domain])
        else:
            bool_index = np.full_like(a_domain, False, dtype=bool)
            for domain in domains:
                bool_index = np.logical_or(a_domain == domain, bool_index)
            c0 = visual_matrix[bool_index, 0]
            c1 = visual_matrix[bool_index, 1]
            a_description = annotations[bool_index, 1]
            a_colors = np.array([domain_colormap[domain_name]
                                 for domain_name in a_domain[bool_index]])

        if keywords is not None:
            bool_list = []
            for description in a_description:
                flag = False
                string = description.lower()
                for keyword in keywords:
                    if keyword in string:
                        flag = True
                        break
                bool_list.append(flag)
            bool_index = np.array(bool_list)
            c0 = c0[bool_index]
            c1 = c1[bool_index]
            a_description = a_description[bool_index]
            a_colors = a_colors[bool_index]

        source = ColumnDataSource(data={
            "c0": c0,
            "c1": c1,
            "color": a_colors,
            "desc": a_description,
        })
        tooltips = [("item", "@desc")]

        plot = figure(
            title=title,
            x_axis_label="c0",
            y_axis_label="c1",
            tooltips=tooltips
        )
        plot.circle("c0", "c1", color="color", fill_alpha=0.2, size=8, source=source)

        return plot

    def show_pyldavis(self, mds="pcoa"):
        """Show interactive visual mapping of LDA topics with pyLDAvis.
        This method is assumed used in jupyter notebook.
        (pyLDAvis currently not working in jupyter lab.)"""
        return self._loglda.show_pyldavis(mds=mds)

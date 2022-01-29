import os
from collections import defaultdict
from itertools import combinations
from typing import Optional, Dict, List, Any

import numpy as np
from sklearn.cluster import DBSCAN

from .lda.base import LogTopicModel

DBSCAN_EPS_CANDIDATES = [
    0.01, 0.03, 0.05, 0.07,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0
]


class TopicClustering(object):

    def __init__(self, model="gensim",
                 knowledge_sources=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False,
                 random_seed=None,
                 lda_n_topics=None,
                 lda_use_cache=False,
                 lda_cachename="ldamodel",
                 verbose=False):
        self._model = model
        self._use_nltk_stopwords = use_nltk_stopwords
        self._use_sklearn_stopwords = use_sklearn_stopwords
        self._random_seed = random_seed
        self._given_n_topics = lda_n_topics
        self._use_ldacache = lda_use_cache
        self._cachename = lda_cachename
        self._verbose = verbose

        if knowledge_sources is None:
            self._knowledge_sources = ["self", ]
        else:
            self._knowledge_sources = knowledge_sources
        if stop_words is None:
            self._stop_words = []
        else:
            self._stop_words = stop_words

        # local variables for rule-based parameter tuning
        self._tuning_rules: Optional[Dict[str, Any]] = None

        # local variables used after fit
        self._loglda: Optional[LogTopicModel] = None
        # _topic_matrix.shape: (len(documents), n_topics)
        self._topic_matrix: Optional[np.array] = None
        self._cluster_labels: Optional[List[int]] = None
        self._clusters: Optional[Dict[int, List[int]]] = None

    @property
    def clusters(self):
        if self._clusters is None:
            raise ValueError("Try fit() first")
        return self._clusters

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            raise ValueError("Try fit() first")
        return self._cluster_labels

    # @staticmethod
    # def _clusters_to_labels(clusters):
    #     ret = [-1] * sum(len(cluster) for cluster in clusters.values())
    #     for cls, l_idx in clusters.items():
    #         for idx in l_idx:
    #             ret[idx] = cls
    #     assert -1 not in ret
    #     return ret

    def _lda_instance(self, documents):
        if self._model == "gensim":
            from .lda import LogLDAgensim
            return LogLDAgensim(
                documents,
                random_seed=self._random_seed,
                stop_words=self._stop_words,
                use_nltk_stopwords=self._use_nltk_stopwords,
                use_sklearn_stopwords=self._use_sklearn_stopwords
            )
        elif self._model == "gibbs":
            raise NotImplementedError
        else:
            raise ValueError

    @staticmethod
    def _lda_cache_filename(name, n_topics, random_seed):
        return "{0}_{1}_{2}".format(name, n_topics, random_seed)

    def _make_lda(self, loglda, n_topics, use_cache=None, verbose=False):
        # fit lda
        # instead use cache if exists
        if use_cache is None:
            use_cache = self._use_ldacache
        name = self._cachename
        filepath = self._lda_cache_filename(name, n_topics, self._random_seed)

        if use_cache:
            if os.path.exists(filepath):
                if verbose:
                    print("use ldamodel cache {0}".format(filepath))
                loglda.load(filepath)
            else:
                loglda.fit(n_topics=n_topics)
                loglda.dump(filepath)
                if verbose:
                    print("output ldamodel cache {0}".format(filepath))
        else:
            loglda.fit(n_topics=n_topics)

        return loglda

    @staticmethod
    def _get_n_topic_candidates(n_documents):
        import math
        max_topics = min(int(math.sqrt(n_documents)) * 2, 60)
        return list(range(10, max_topics, 10))

    def lda_search_param(self, training_docs):
        n_topic_candidates = self._get_n_topic_candidates(len(training_docs))
        loglda = self._lda_instance(training_docs)

        for n_topics in n_topic_candidates:
            loglda = self._make_lda(loglda, n_topics)
            corpus = loglda.corpus(training_docs)
            lp = loglda.log_perplexity(corpus=corpus)
            coherence = loglda.coherence(corpus=corpus)
            yield [n_topics, lp, coherence]

    @staticmethod
    def _split_clusters(clustering_labels, opr_outlier="last"):
        # outliers will be an additional cluster
        outliers = []
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering_labels):
            if label == -1:
                outliers.append(idx)
            else:
                clusters[label].append(idx)

        if len(clusters) == 0:
            raise ValueError("Clustering failed")

        if opr_outlier == "last":
            # 1 cluster for all outliers as the last cluster
            clusters[max(clusters.keys()) + 1] = outliers
        elif opr_outlier == "outlier":
            # 1 cluster for all outliers as -1th cluster
            clusters[-1] = outliers
        elif opr_outlier == "scatter":
            # 1 cluster for 1 outlier data
            key = max(clusters.keys()) + 1
            for idx in outliers:
                clusters[key] = [idx]
                key = key + 1
        elif opr_outlier == "ignore":
            # ignore outliers in the results
            pass
        else:
            raise NotImplementedError

        return clusters

    def test_tuning_rules(self, loglda, clusters, topic_matrix):
        topn = self._tuning_rules["topn"]
        term_class = self._tuning_rules["term_class"]
        if term_class == "topic":
            tested_terms = loglda.all_topic_terms(topn=topn)
        elif term_class == "cluster":
            tested_terms = self._all_cluster_terms(loglda, clusters,
                                                   topic_matrix, topn=topn)
        else:
            raise ValueError

        union_checker = {}
        for rule in self._tuning_rules["union"]:
            for w1, w2 in combinations(rule, 2):
                union_checker[(w1, w2)] = True
        separation_checker = {}
        for rule in self._tuning_rules["separation"]:
            for w1, w2 in combinations(rule, 2):
                separation_checker[(w1, w2)] = True
        score_denom = len(union_checker) + len(separation_checker)
        assert score_denom > 0, "bad tuning rule specification"

        for clsid, bestn in tested_terms.items():
            s_words = set(w for w, _ in bestn)
            for w1, w2 in union_checker:
                # violate if w1 and w2 are not always in 1 class terms
                if w1 in s_words != w2 in s_words:
                    union_checker[(w1, w2)] = False
            for w1, w2 in separation_checker:
                # violate if w1 and w2 are in 1 class terms
                if w1 in s_words and w2 in s_words:
                    separation_checker[(w1, w2)] = False

        score = sum(1 for _, v in union_checker if v)
        score += sum(1 for _, v in separation_checker if v)
        return score / score_denom

    def set_tuning_rules(self, union_rules: List[List[str]],
                         separation_rules: List[List[str]],
                         term_class="topic", topn=10):
        self._tuning_rules = {"union": union_rules,
                              "separation": separation_rules,
                              "term_class": term_class,
                              "topn": topn}

    @staticmethod
    def _get_cluster_centers(clusters, topic_matrix):
        l_avg_v = []
        for cid, cluster in clusters.items():
            l_cls_topicv = [topic_matrix[idx] for idx in cluster]
            avg_v = np.array(l_cls_topicv).mean(axis=0)
            l_avg_v.append(avg_v)
        return np.array(l_avg_v)

    def get_cluster_centers(self):
        return self._get_cluster_centers(self._clusters, self._topic_matrix)

    def cluster_terms(self, cluster_id, topn=10):
        from gensim import matutils
        cluster_voc_matrix = np.dot(self.get_cluster_centers(),
                                    self._loglda.get_topics())
        topicv = cluster_voc_matrix[cluster_id]
        topicv = topicv / topicv.sum()
        bestn = matutils.argsort(topicv, topn, reverse=True)
        return [(self._loglda.id2word(widx), topicv[widx])
                for widx in bestn]

    @classmethod
    def _all_cluster_terms(cls, loglda, clusters, topic_matrix,
                           topn=10):
        from gensim import matutils
        cluster_voc_matrix = np.dot(
            cls._get_cluster_centers(clusters, topic_matrix),
            loglda.get_topics()
        )
        ret = {}
        for cid in clusters:
            topicv = cluster_voc_matrix[cid]
            topicv = topicv / topicv.sum()
            bestn = matutils.argsort(topicv, topn, reverse=True)
            ret[cid] = [(loglda.id2word(widx), topicv[widx])
                        for widx in bestn]
        return ret

    def all_cluster_terms(self, topn=10):
        return self._all_cluster_terms(
            self._loglda, self._clusters, self._topic_matrix, topn=topn
        )

    def inspection(self, topn=10):
        return {
            "topic_terms": self._loglda.all_topic_terms(topn),
            "cluster_terms": self.all_cluster_terms(topn),
        }

    def show_pyldavis(self, mds="pcoa"):
        return self._loglda.show_pyldavis(mds=mds)

    def get_principal_components(self, input_docs, n_components=2,
                                 use_zscore=False):
        corpus = self._loglda.corpus(input_docs)
        return self._loglda.principal_components(
            corpus,
            n_components=n_components,
            use_zscore=use_zscore
        )


class TopicClusteringDBSCAN(TopicClustering):

    def __init__(self, model="gensim",
                 knowledge_sources=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False,
                 random_seed=None,
                 lda_n_topics=None,
                 lda_use_cache=False,
                 lda_cachename="ldamodel",
                 redistribute=True,
                 cluster_eps=None,
                 cluster_size_min=5,
                 tuning_metrics="cluster",
                 verbose=False):

        super().__init__(model=model,
                         knowledge_sources=knowledge_sources,
                         stop_words=stop_words,
                         use_nltk_stopwords=use_nltk_stopwords,
                         use_sklearn_stopwords=use_sklearn_stopwords,
                         random_seed=random_seed,
                         lda_n_topics=lda_n_topics,
                         lda_use_cache=lda_use_cache,
                         lda_cachename=lda_cachename,
                         verbose=verbose)
        self._redistribute = redistribute
        self._given_cluster_eps = cluster_eps
        self._min_samples = cluster_size_min
        self._tuning_metrics = tuning_metrics

        # local variables used after fit
        self._eps: Optional[float] = None

    @staticmethod
    def _get_eps_candidates():
        return DBSCAN_EPS_CANDIDATES

    def _tuning_metrics_score(self, clustering_labels, verbose=False):
        if self._tuning_metrics == "cluster":
            # max clusters
            return len(set(clustering_labels))
        elif self._tuning_metrics == "outlier":
            # minimum outliers
            n_docs = len(clustering_labels)
            n_outlier = sum(1 for label in clustering_labels
                            if label == -1)
            return (n_docs - n_outlier) / n_docs
        elif self._tuning_metrics == "both":
            n_max_cluster = len(set(clustering_labels))
            n_docs = len(clustering_labels)
            n_outlier = sum(1 for label in clustering_labels
                            if label == -1)
            outlier_ratio = n_outlier / n_docs
            score = n_max_cluster ** 2 * (n_docs - n_outlier) / n_docs
            if verbose:
                print("max_cluster: {0}, outlier ratio: {1} -> {2}".format(
                    n_max_cluster, outlier_ratio, score
                ))
            return score

    @classmethod
    def _do_clustering(cls, topic_matrix, **dbscan_kwargs):
        clustering = DBSCAN(**dbscan_kwargs)
        clustering.fit(topic_matrix)
        labels = clustering.labels_.copy()
        clusters = cls._split_clusters(labels, opr_outlier="last")
        return labels, clusters

    def param_tuning(self, training_docs, input_docs,
                     min_samples=None, verbose=False):
        # tune two parameters: LDA n_topic and DBSCAN eps
        if min_samples is None:
            min_samples = self._min_samples

        if self._given_n_topics is None:
            n_topic_candidates = self._get_n_topic_candidates(len(training_docs))
        else:
            n_topic_candidates = [self._given_n_topics, ]

        if self._given_cluster_eps is None:
            eps_candidates = self._get_eps_candidates()
        else:
            eps_candidates = [self._given_cluster_eps, ]

        loglda = self._lda_instance(training_docs)
        input_corpus = loglda.corpus(input_docs)

        d_metric_scores = {}
        d_rule_scores = {}
        for n_topics in n_topic_candidates:
            loglda = self._make_lda(loglda, n_topics, verbose=verbose)
            topic_matrix = loglda.corpus_topic_matrix(
                input_corpus, use_zscore=True,
            )

            for eps in eps_candidates:
                dbscan_kwargs = {"eps": eps,
                                 "min_samples": min_samples,
                                 "metric": "cityblock"}
                labels, clusters = self._do_clustering(
                    topic_matrix, **dbscan_kwargs
                )

                params = (n_topics, eps)
                d_metric_scores[params] = self._tuning_metrics_score(
                    labels, verbose=verbose
                )
                if self._tuning_rules:
                    if self._redistribute:
                        labels, clusters = self._split_and_redistribute_clusters(
                            labels, topic_matrix
                        )
                    else:
                        labels, clusters = self._split_clusters(
                            labels, opr_outlier="last"
                        )
                    score = self.test_tuning_rules(loglda, clusters,
                                                   topic_matrix)
                    d_rule_scores[params] = score
                    if verbose:
                        print(params, d_rule_scores[params], d_metric_scores[params])
                else:
                    if verbose:
                        print(params, d_metric_scores[params])

        if self._tuning_rules:
            # keep params that have maximum score
            max_score = max(d_rule_scores.values())
            tmp = {}
            for params, score in d_rule_scores.items():
                if score == max_score:
                    tmp[params] = d_metric_scores[params]
            d_metric_scores = tmp

        # get params with maximum scores
        selected_param, _ = sorted(d_metric_scores.items(),
                                   key=lambda x: x[1], reverse=True)[0]
        if verbose:
            print("Selected {0}".format(selected_param))
        return selected_param

    @staticmethod
    def _split_and_redistribute_clusters(cluster_labels, topic_matrix):
        # split clusters and outliers
        outliers = []
        new_clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label == -1:
                outliers.append(idx)
            else:
                new_clusters[label].append(idx)

        if len(new_clusters) == 0:
            raise ValueError("Clustering failed")

        # average distribution (centers) of clusters
        l_cluster_centers = []
        for cid, cluster in new_clusters.items():
            l_distance = [topic_matrix[idx] for idx in cluster]
            avg = np.array(l_distance).mean(axis=0)
            l_cluster_centers.append(avg)

        # re-distribute outliers to nearest clusters
        from scipy.spatial.distance import cityblock
        for outlier in outliers:
            outlier_v = topic_matrix[outlier]
            l_distance = [(cid, cityblock(center_v, outlier_v))
                          for cid, center_v in enumerate(l_cluster_centers)]
            min_cls, min_distance = min(l_distance, key=lambda x: x[1])
            new_clusters[min_cls].append(outlier)

        new_labels = np.array([-1] * len(cluster_labels))
        for cid, l_idx in new_clusters.items():
            new_labels[np.array(l_idx)] = cid
        assert -1 not in new_labels

        return new_labels, new_clusters

    def fit(self, input_docs, training_docs=None):
        if training_docs is None:
            training_docs = input_docs
        if self._given_n_topics is None or self._given_cluster_eps is None:
            n_topics, eps = self.param_tuning(
                training_docs, input_docs,
                min_samples=self._min_samples,
                verbose=self._verbose
            )
        else:
            n_topics = self._given_n_topics
            eps = self._given_cluster_eps

        self._loglda = self._lda_instance(training_docs)
        self._loglda = self._make_lda(self._loglda, n_topics, verbose=self._verbose)
        input_corpus = self._loglda.corpus(input_docs)
        self._topic_matrix = self._loglda.corpus_topic_matrix(
            input_corpus, use_zscore=True
        )

        dbscan_kwargs = {"eps": eps,
                         "min_samples": self._min_samples,
                         "metric": "cityblock"}
        self._cluster_labels, self._clusters = self._do_clustering(
            self._topic_matrix, **dbscan_kwargs
        )
        if self._redistribute:
            self._cluster_labels, self._clusters = self._split_and_redistribute_clusters(
                self._cluster_labels, self._topic_matrix
            )

        return self._clusters


class TopicClusteringRecursiveDBSCAN(TopicClustering):

    def __init__(self, model="gensim",
                 knowledge_sources=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False,
                 random_seed=None,
                 lda_n_topics=None,
                 lda_use_cache=False,
                 lda_cachename="ldamodel",
                 cluster_eps=None,
                 recursive_cluster_size=20,
                 verbose=False):

        super().__init__(model=model,
                         knowledge_sources=knowledge_sources,
                         stop_words=stop_words,
                         use_nltk_stopwords=use_nltk_stopwords,
                         use_sklearn_stopwords=use_sklearn_stopwords,
                         random_seed=random_seed,
                         lda_n_topics=lda_n_topics,
                         lda_use_cache=lda_use_cache,
                         lda_cachename=lda_cachename,
                         verbose=verbose)
        self._given_cluster_eps = cluster_eps
        self._recursive_cluster_size = recursive_cluster_size

        # local variables used after fit
        self._eps: Optional[float] = None

    @staticmethod
    def _get_eps_candidates():
        return DBSCAN_EPS_CANDIDATES

    def lda_search_param(self, training_docs):
        n_topic_candidates = self._get_n_topic_candidates(len(training_docs))
        loglda = self._lda_instance(training_docs)

        for n_topics in n_topic_candidates:
            loglda.fit(n_topics=n_topics)
            corpus = loglda.corpus(training_docs)
            lp = loglda.log_perplexity(corpus=corpus)
            coherence = loglda.coherence(corpus=corpus)
            yield [n_topics, lp, coherence]

    @classmethod
    def _do_clustering(cls, topic_matrix, a_idx, cluster_size,
                       **dbscan_kwargs):
        clustering = DBSCAN(**dbscan_kwargs)
        clustering.fit(topic_matrix[a_idx, :])
        labels = clustering.labels_.copy()
        clusters = cls._split_clusters(labels, opr_outlier="scatter")

        # avoid infinite loop
        if len(clusters) == 1:
            return labels, clusters

        # local recursive clustering in larger clusters
        updated_clusters = {}
        for cid, l_elm in clusters.items():
            if len(l_elm) > cluster_size:
                l_idx = [a_idx[elm] for elm in l_elm]
                new_a_idx = np.array(l_idx)
                new_labels, new_clusters = cls._do_clustering(
                    topic_matrix, new_a_idx,
                    cluster_size,
                    **dbscan_kwargs
                )
                if len(new_clusters) > 1:
                    updated_clusters[cid] = new_clusters

        # update labels and clusters by recursive results
        for cid, tmp_clusters in updated_clusters.items():
            n_clusters = len(clusters)
            new_cids = [cid] + list(range(n_clusters, n_clusters + len(tmp_clusters) - 1))
            clusters.pop(cid)
            for new_cid, tmp_cluster in zip(new_cids, tmp_clusters.values()):
                clusters[new_cid] = tmp_cluster
                labels[np.array(tmp_cluster)] = new_cid

        if len(updated_clusters) == 0:
            print("no recursion")

        # sanity check
        test_clusters = cls._split_clusters(labels, opr_outlier="scatter")
        for cid in clusters:
            assert clusters[cid] == test_clusters[cid]

        return labels, clusters

    def param_tuning(self, training_docs, input_docs,
                     verbose=False):
        # tune two parameters: LDA n_topic and DBSCAN eps
        if self._given_n_topics is None:
            n_topic_candidates = self._get_n_topic_candidates(len(training_docs))
        else:
            n_topic_candidates = [self._given_n_topics, ]

        if self._given_cluster_eps is None:
            eps_candidates = self._get_eps_candidates()
        else:
            eps_candidates = [self._given_cluster_eps, ]

        loglda = self._lda_instance(training_docs)
        input_corpus = loglda.corpus(input_docs)

        d_metric_scores = {}
        d_rule_scores = {}
        for n_topics in n_topic_candidates:
            self._loglda = self._make_lda(self._loglda, n_topics, verbose=verbose)
            topic_matrix = loglda.corpus_topic_matrix(
                input_corpus, use_zscore=True
            )

            for eps in eps_candidates:
                dbscan_kwargs = {"eps": eps,
                                 "min_samples": 1,
                                 "metric": "cityblock"}
                initial_a_idx = np.array(range(len(input_corpus)))
                labels, clusters = self._do_clustering(
                    topic_matrix, initial_a_idx, self._recursive_cluster_size,
                    **dbscan_kwargs
                )

                params = (n_topics, eps)
                d_metric_scores[params] = len(input_corpus) / len(clusters)

                if self._tuning_rules:
                    score = self.test_tuning_rules(loglda, clusters,
                                                   topic_matrix)
                    d_rule_scores[params] = score
                    if verbose:
                        print(params, d_rule_scores[params], d_metric_scores[params])
                else:
                    if verbose:
                        print(params, d_metric_scores[params])

        if self._tuning_rules:
            # keep params that have maximum score
            max_score = max(d_rule_scores.values())
            tmp = {}
            for params, score in d_rule_scores.items():
                if score == max_score:
                    tmp[params] = d_metric_scores[params]
            d_metric_scores = tmp

        # get params with maximum scores
        selected_param, _ = sorted(d_metric_scores.items(),
                                   key=lambda x: x[1], reverse=True)[0]
        if verbose:
            print("Selected {0}".format(selected_param))
        return selected_param

    def fit(self, input_docs, training_docs=None):
        if training_docs is None:
            training_docs = input_docs
        if self._given_n_topics is None or self._given_cluster_eps is None:
            n_topics, eps = self.param_tuning(
                training_docs, input_docs,
                verbose=self._verbose
            )
        else:
            n_topics = self._given_n_topics
            eps = self._given_cluster_eps

        self._loglda = self._lda_instance(training_docs)
        self._loglda = self._make_lda(self._loglda, n_topics, verbose=self._verbose)
        input_corpus = self._loglda.corpus(input_docs)
        self._topic_matrix = self._loglda.corpus_topic_matrix(
            input_corpus, use_zscore=True
        )

        dbscan_kwargs = {"eps": eps,
                         "min_samples": 1,
                         "metric": "cityblock"}
        initial_a_idx = np.array(range(len(input_corpus)))
        self._cluster_labels, self._clusters = self._do_clustering(
            self._topic_matrix, initial_a_idx, self._recursive_cluster_size,
            **dbscan_kwargs
        )

        return self._clusters

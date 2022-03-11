from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

DBSCAN_EPS_CANDIDATES = [
    0.01, 0.03, 0.05, 0.07,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0
]


class TopicClustering(ABC):

    def __init__(self):
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

    @abstractmethod
    def need_tuning(self):
        raise NotImplementedError

    @abstractmethod
    def set_param(self, params):
        raise NotImplementedError

    @abstractmethod
    def get_kwargs(self):
        raise NotImplementedError

    @abstractmethod
    def get_param_candidates(self):
        """Returns key parameter (for tuning) and clustering parameters (kwargs)"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tuning_metrics(cls, clustering_labels, metrics_name, verbose=False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def compare_metrics(cls, metrics1, metrics2, metrics_name):
        raise NotImplementedError

    @abstractmethod
    def clustering(self, topic_matrix, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, topic_matrix):
        raise NotImplementedError

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

    def cluster_terms(self, loglda, cluster_id, topn=10):
        from gensim import matutils
        cluster_voc_matrix = np.dot(self.get_cluster_centers(),
                                    loglda.get_topics())
        topicv = cluster_voc_matrix[cluster_id]
        topicv = topicv / topicv.sum()
        bestn = matutils.argsort(topicv, topn, reverse=True)
        return [(loglda.id2word(widx), topicv[widx])
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
            topicv_sum = topicv.sum()
            topicv = np.divide(
                topicv, topicv_sum, out=np.zeros_like(topicv),
                where=(topicv_sum != 0)
            )
            bestn = matutils.argsort(topicv, topn, reverse=True)
            ret[cid] = [(loglda.id2word(widx), topicv[widx])
                        for widx in bestn]
        return ret

    def all_cluster_terms(self, loglda, topn=10):
        return self._all_cluster_terms(
            loglda, self._clusters, self._topic_matrix, topn=topn
        )


class TopicClusteringDBSCAN(TopicClustering):

    def __init__(self,
                 dbscan_eps=None,
                 cluster_size_min=5,
                 redistribute=True
                 ):

        super().__init__()
        self._dbscan_eps = dbscan_eps
        self._min_samples = cluster_size_min
        self._redistribute = redistribute

    def _clustering_kwargs(self, eps):
        return {
            "eps": eps,
            "min_samples": self._min_samples,
            "metric": "cityblock"
        }

    def need_tuning(self):
        return self._dbscan_eps is None

    def set_param(self, params):
        self._dbscan_eps = params

    def get_kwargs(self):
        assert self._dbscan_eps is not None
        return self._clustering_kwargs(self._dbscan_eps)

    def get_param_candidates(self):
        for eps in DBSCAN_EPS_CANDIDATES:
            yield eps, self._clustering_kwargs(eps)

    @classmethod
    def tuning_metrics(cls, clustering_labels, metrics_name, verbose=False):
        if metrics_name == "cluster":
            # max clusters
            return len(set(clustering_labels))
        elif metrics_name == "outlier":
            # minimum outliers
            n_docs = len(clustering_labels)
            n_outlier = sum(1 for label in clustering_labels
                            if label == -1)
            return (n_docs - n_outlier) / n_docs
        elif metrics_name == "both":
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
        else:
            raise ValueError

    @classmethod
    def compare_metrics(cls, metrics1, metrics2, metrics_name):
        """Return True if metrics2 is better than metrics1."""
        if metrics_name == "cluster":
            return metrics1 < metrics2
        elif metrics_name == "outlier":
            return metrics1 < metrics2
        elif metrics_name == "both":
            return metrics1 < metrics2
        else:
            raise ValueError

    def clustering(self, topic_matrix, **kwargs):
        clustering = DBSCAN(**kwargs)
        clustering.fit(topic_matrix)
        labels = clustering.labels_.copy()
        clusters = self._split_clusters(labels, opr_outlier="last")
        return labels, clusters

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

    def fit(self, topic_matrix):
        assert self._dbscan_eps is not None

        self._topic_matrix = topic_matrix

        kwargs = self._clustering_kwargs(self._dbscan_eps)
        self._cluster_labels, self._clusters = self.clustering(
            self._topic_matrix, **kwargs
        )
        if self._redistribute:
            self._cluster_labels, self._clusters = self._split_and_redistribute_clusters(
                self._cluster_labels, self._topic_matrix
            )

        return self._clusters


class TopicClusteringRecursiveDBSCAN(TopicClustering):

    def __init__(self,
                 dbscan_eps=None,
                 recursive_cluster_size=20
                 ):

        super().__init__()
        self._dbscan_eps = dbscan_eps
        self._recursive_cluster_size = recursive_cluster_size

    @staticmethod
    def _clustering_kwargs(eps):
        return {
            "eps": eps,
            "min_samples": 1,
            "metric": "cityblock"
        }

    def need_tuning(self):
        return self._dbscan_eps is None

    def set_param(self, params):
        self._dbscan_eps = params

    def get_kwargs(self):
        assert self._dbscan_eps is not None
        return self._clustering_kwargs(self._dbscan_eps)

    def get_param_candidates(self):
        for eps in DBSCAN_EPS_CANDIDATES:
            yield eps, self._clustering_kwargs(eps)

    @classmethod
    def tuning_metrics(cls, clustering_labels, metrics_name, verbose=False):
        if metrics_name == "cluster":
            return len(set(clustering_labels))
        else:
            raise ValueError

    @classmethod
    def compare_metrics(cls, metrics1, metrics2, metrics_name):
        """Return True if metrics2 is better than metrics1."""
        if metrics_name == "cluster":
            # minimize n_cluster
            return metrics1 > metrics2
        else:
            raise ValueError

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

    def clustering(self, topic_matrix, **kwargs):
        initial_a_idx = np.array(range(len(topic_matrix)))
        return self._do_clustering(
            topic_matrix,
            initial_a_idx,
            self._recursive_cluster_size,
            **kwargs
        )

    def fit(self, topic_matrix):
        assert self._dbscan_eps is not None

        self._topic_matrix = topic_matrix

        kwargs = self._clustering_kwargs(self._dbscan_eps)
        self._cluster_labels, self._clusters = self.clustering(
            self._topic_matrix, **kwargs
        )

        return self._clusters


class ParameterTuning:

    def __init__(self,
                 tuning_metrics="cluster",
                 cache_dir=None,
                 lda_use_zscore=True,
                 lda_use_cache=True,
                 lda_cachename=None,
                 ):
        self._tuning_metrics = tuning_metrics
        self._cache_dir = cache_dir
        self._lda_use_zscore = lda_use_zscore,
        self._lda_use_cache = lda_use_cache,
        self._lda_cachename = lda_cachename,

        self._tuning_rules = None

    @staticmethod
    def _get_n_topics_candidates(n_documents):
        import math
        max_topics = min(int(math.sqrt(n_documents)) * 2, 60)
        return list(range(10, max_topics, 10))

    @classmethod
    def get_lda_param_candidates(cls, documents):
        for n_topics in cls._get_n_topics_candidates(len(documents)):
            yield n_topics

    def set_tuning_rules(self, union_rules: List[List[str]],
                         separation_rules: List[List[str]],
                         term_class="topic", topn=10):
        """Rule-based tuning"""
        self._tuning_rules = {"union": union_rules,
                              "separation": separation_rules,
                              "term_class": term_class,
                              "topn": topn}

    def test_tuning_rules(self, loglda, clusters, topic_matrix):
        topn = self._tuning_rules["topn"]
        term_class = self._tuning_rules["term_class"]
        if term_class == "topic":
            tested_terms = loglda.all_topic_terms(topn=topn)
        elif term_class == "cluster":
            tested_terms = TopicClustering._all_cluster_terms(
                loglda, clusters, topic_matrix, topn=topn
            )
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

    def fit_or_load_loglda(self, loglda, training_docs, n_topics,
                           verbose=False):
        if self._lda_use_cache:
            if loglda.cache_exists(self._lda_cachename):
                loglda.load(self._lda_cachename, self._cache_dir)
                if verbose:
                    print("load LDA cache ({0} in {1})".format(
                        self._lda_cachename, self._cache_dir)
                    )
            else:
                loglda.fit(training_docs, n_topics=n_topics)
                loglda.dump(self._lda_cachename, self._cache_dir)
                if verbose:
                    print("dump LDA cache ({0} in {1})".format(
                        self._lda_cachename, self._cache_dir)
                    )
        else:
            loglda.fit(training_docs, n_topics=n_topics)
        return loglda

    def _measure_scores(self, loglda, tcls, topic_matrix, tcls_kwargs,
                        verbose):
        labels, clusters = tcls.clustering(
            topic_matrix,
            **tcls_kwargs
        )

        metrics = tcls.tuning_metrics(
            labels, metrics_name=self._tuning_metrics, verbose=verbose
        )

        if self._tuning_rules:
            rule_score = self.test_tuning_rules(
                loglda, clusters, topic_matrix
            )
        else:
            rule_score = None

        return metrics, rule_score

    def _select_params(self, tcls, d_metrics, d_rule_scores):
        if self._tuning_rules:
            # keep params in d_metrics that have maximum score
            max_score = max(d_rule_scores.values())
            tmp = {}
            for params_key, score in d_rule_scores.items():
                if score == max_score:
                    tmp[params_key] = d_metrics[params_key]
            d_metrics = tmp

        # get params with best scores
        max_params_key = None
        max_metrics = None
        for params_key, metrics in d_metrics.items():
            if max_params_key is None:
                max_params_key = params_key
                max_metrics = metrics
            elif tcls.compare_metrics(max_metrics, metrics, self._tuning_metrics):
                max_params_key = params_key
                max_metrics = metrics

        return max_params_key

    def tune_lda(self, loglda, tcls, training_docs, input_docs, verbose=False):
        if verbose:
            print("tuning LDA parameters (n_topics)")

        original_n_topics = loglda.n_topics
        loglda_param_sets = self.get_lda_param_candidates(training_docs)
        tcls_kwargs = tcls.get_kwargs()

        d_metrics = {}
        d_rule_scores = {}

        for n_topics in loglda_param_sets:
            loglda.set_n_topics(n_topics)
            loglda = self.fit_or_load_loglda(
                loglda, training_docs, n_topics, verbose=verbose
            )
            input_corpus = loglda.corpus(input_docs)
            topic_matrix = loglda.topic_matrix(
                input_corpus, use_zscore=self._lda_use_zscore
            )

            metrics, rule_score = self._measure_scores(
                loglda, tcls, topic_matrix, tcls_kwargs, verbose
            )
            params_key = n_topics
            d_metrics[params_key] = metrics
            d_rule_scores[params_key] = rule_score
            if verbose:
                if self._tuning_rules:
                    print("params: {0}, rule_score: {1}, metrics: {2}".format(
                        params_key, rule_score, metrics)
                    )
                else:
                    print("params: {0}, metrics: {1}".format(params_key, metrics))

        loglda.set_n_topics(original_n_topics)

        max_params_key = self._select_params(tcls, d_metrics, d_rule_scores)
        if verbose:
            print("Selected {0}".format(max_params_key))
        return max_params_key

    def tune_clustering(self, loglda, tcls, input_docs, verbose=False):
        """Note: loglda should be already trained with loglda.fit"""
        assert loglda.vocabulary()

        if verbose:
            print("tuning clustering parameters")

        tcls_param_sets = tcls.get_param_candidates()
        input_corpus = loglda.corpus(input_docs)
        topic_matrix = loglda.topic_matrix(
            input_corpus, use_zscore=self._lda_use_zscore
        )

        d_metrics = {}
        d_rule_scores = {}
        for tcls_key_param, tcls_kwargs in tcls_param_sets:
            metrics, rule_score = self._measure_scores(
                loglda, tcls, topic_matrix, tcls_kwargs, verbose
            )
            params_key = tcls_key_param
            d_metrics[params_key] = metrics
            d_rule_scores[params_key] = rule_score
            if verbose:
                if self._tuning_rules:
                    print("params: {0}, rule_score: {1}, metrics: {2}".format(
                        params_key, rule_score, metrics)
                    )
                else:
                    print("params: {0}, metrics: {1}".format(params_key, metrics))

        max_params_key = self._select_params(tcls, d_metrics, d_rule_scores)
        if verbose:
            print("Selected {0}".format(max_params_key))
        return max_params_key

    def tune_all(self, loglda, tcls, training_docs, input_docs, verbose=False):
        if verbose:
            print("tuning parameters for both LDA and clustering")

        original_n_topics = loglda.n_topics
        loglda_param_sets = self.get_lda_param_candidates(training_docs)
        tcls_param_sets = tcls.get_param_candidates()

        d_metrics = {}
        d_rule_scores = {}
        for n_topics in loglda_param_sets:
            loglda.set_n_topics(n_topics)
            loglda = self.fit_or_load_loglda(
                loglda, training_docs, n_topics, verbose=verbose
            )
            input_corpus = loglda.corpus(input_docs)
            topic_matrix = loglda.topic_matrix(
                input_corpus, use_zscore=self._lda_use_zscore
            )

            for tcls_key_param, tcls_kwargs in tcls_param_sets:
                metrics, rule_score = self._measure_scores(
                    loglda, tcls, topic_matrix, tcls_kwargs, verbose
                )
                params_key = tcls_key_param
                d_metrics[params_key] = metrics
                d_rule_scores[params_key] = rule_score
                if verbose:
                    if self._tuning_rules:
                        print("params: {0}, rule_score: {1}, metrics: {2}".format(
                            params_key, rule_score, metrics)
                        )
                    else:
                        print("params: {0}, metrics: {1}".format(params_key, metrics))

        loglda.set_n_topics(original_n_topics)

        max_params_key = self._select_params(tcls, d_metrics, d_rule_scores)
        if verbose:
            print("Selected {0}".format(max_params_key))
        return max_params_key

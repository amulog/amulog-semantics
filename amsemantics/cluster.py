from collections import defaultdict
from itertools import combinations
from typing import Optional, Dict, List, Any

import numpy as np
from sklearn.cluster import DBSCAN

from .lda.base import LogTopicModel


class TopicClustering:

    def __init__(self, model="gensim", redistribute=True,
                 stop_words=None, random_seed=None, verbose=False):
        self._model = model
        self._redistribute = redistribute
        self._random_seed = random_seed
        self._verbose = verbose
        if stop_words is None:
            self._stop_words = []
        else:
            self._stop_words = stop_words

        # local variables for parameter tuning
        self._tuning_rules: Optional[Dict[str, Any]] = None

        # local variables used after fit
        self._loglda: Optional[LogTopicModel] = None
        # _topic_matrix.shape: (len(documents), n_topics)
        self._topic_matrix: Optional[np.array] = None
        self._eps: Optional[float] = None
        self._clustering: Optional[DBSCAN] = None
        self._clusters: Optional[Dict[int, List[int]]] = None

    def _lda_instance(self, documents):
        if self._model == "gensim":
            from .lda import LogLDAgensim
            return LogLDAgensim(
                documents,
                stop_words=self._stop_words,
                random_seed=self._random_seed
            )
        elif self._model == "gibbs":
            raise NotImplementedError
        else:
            raise ValueError

    @staticmethod
    def _get_n_topic_candidates(n_documents):
        import math
        return list(range(10, int(math.sqrt(n_documents)) * 2, 10))

    @staticmethod
    def _get_eps_candidates():
        return [0.01, 0.03, 0.05, 0.07,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

    def param_tuning(self, documents, tuning_lda=True,
                     min_samples=5, verbose=False):
        if tuning_lda:
            n_topic_candidates = self._get_n_topic_candidates(len(documents))
        else:
            n_topic_candidates = [40, ]

        loglda = self._lda_instance(documents)

        d_n_clusters = {}
        d_scores = {}
        for n_topics in n_topic_candidates:
            loglda.fit(n_topics=n_topics)
            topic_matrix = loglda.corpus_topic_matrix()

            for eps in self._get_eps_candidates():
                clustering = DBSCAN(eps=eps, min_samples=min_samples,
                                    metric="cityblock")
                clustering.fit(topic_matrix)

                params = (n_topics, eps)
                d_n_clusters[params] = len(set(clustering.labels_))
                if self._tuning_rules:
                    score = self.test_tuning_rules(loglda, clustering,
                                                   topic_matrix)
                    d_scores[params] = score
                    if verbose:
                        print(params, score, d_n_clusters[params])
                else:
                    if verbose:
                        print(params, d_n_clusters[params])

        if self._tuning_rules:
            # keep params that have maximum score
            max_score = max(d_scores.values())
            tmp = {}
            for params, score in d_scores.items():
                if score == max_score:
                    tmp[params] = d_n_clusters[params]
            d_n_clusters = tmp

        # get params with maximum number of clusters
        selected_param, _ = sorted(d_n_clusters.items(),
                                   key=lambda x: x[1], reverse=True)[0]
        if verbose:
            print("Selected {0}".format(selected_param))
        return selected_param

    def test_tuning_rules(self, loglda, clustering, topic_matrix):
        if self._redistribute:
            clusters = self._split_and_redistribute_clusters(
                clustering, topic_matrix
            )
        else:
            clusters = self._split_clusters(self._clustering)

        topn = self._tuning_rules["topn"]
        tested_terms = self._tuning_rules["tested_terms"]
        if self._tuning_rules["tested_terms"] == "topic":
            tested_terms = loglda.corpus_topic_matrix(topn=topn)
        elif self._tuning_rules["tested_terms"] == "cluster":
            tested_terms = self._all_cluster_terms(loglda, clusters,
                                                   topic_matrix, topn=topn)
        union_checker = {}
        for w1, w2 in combinations(self._tuning_rules["union"], 2):
            union_checker[(w1, w2)] = True
        separation_checker = {}
        for w1, w2 in combinations(self._tuning_rules["separation"], 2):
            separation_checker[(w1, w2)] = True

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
        score = score / (len(union_checker) + len(separation_checker))
        return score

    def set_tuning_rules(self, union_rules: List[List[str]],
                         separation_rules: List[List[str]],
                         term_class="topic", topn=10):
        self._tuning_rules = {"union": union_rules,
                              "separation": separation_rules,
                              "term_class": term_class,
                              "topn": topn}

    @staticmethod
    def _split_and_redistribute_clusters(clustering, topic_matrix):
        # split clusters and outliers
        outliers = []
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                outliers.append(idx)
            else:
                clusters[label].append(idx)

        if len(clusters) == 0:
            raise ValueError("Clustering failed")

        # average distribution (centers) of clusters
        l_cluster_centers = []
        for cid, cluster in clusters.items():
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
            clusters[min_cls].append(outlier)

        return clusters

    @staticmethod
    def _split_clusters(clustering):
        # outliers will be an additional cluster
        outliers = []
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                outliers.append(idx)
            else:
                clusters[label].append(idx)

        if len(clusters) == 0:
            raise ValueError("Clustering failed")

        clusters[max(clusters.keys()) + 1] = outliers
        return clusters

    def fit(self, documents):
        n_topics, eps = self.param_tuning(
            documents, tuning_lda=True,
            min_samples=5, verbose=self._verbose
        )

        self._loglda = self._lda_instance(documents)
        self._loglda.fit()
        self._topic_matrix = self._loglda.corpus_topic_matrix()

        from sklearn.cluster import DBSCAN
        self._clustering = DBSCAN(eps=eps, min_samples=5,
                                  metric="cityblock")
        self._clustering.fit(self._topic_matrix)
        if self._redistribute:
            self._clusters = self._split_and_redistribute_clusters(
                self._clustering, self._topic_matrix
            )
        else:
            self._clusters = self._split_clusters(self._clustering)

        return self._clusters

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
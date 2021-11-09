"""
A tool to use RFC documents as the input of semantic analysis.

Author: Ayato Tokubi, Satoru Kobayashi
"""

import os
import re
import pickle
from typing import Optional, List, Iterable
from nltk.corpus import stopwords

from rfcyaml.RFC import RFC, RFCStatus

from ..nlpnorm import Normalizer

RFC_ID_MIN = 1
RFC_ID_MAX = 9999
DEFAULT_STATUS_TO_USE = {
    RFCStatus.PROPOSED_STANDARD,
    RFCStatus.INTERNET_STANDARD,
    RFCStatus.DRAFT_STANDARD
}
DEFAULT_FILTERS = ["lower", "split_symbols", "lemmatize_verbs", "lemmatize_nns"]
DEFAULT_NORMALIZER = Normalizer(DEFAULT_FILTERS)

# for split_with_special_character()
# special characters:
#     / * + \ | --- _ , -> <- ( ) ! # . [ ] < > ; { } : ' " $ =
SEGMENTATION_REGEX = re.compile(
    r'[/*+\\|]|--+|_|,|->|<-|\(|\)|!|#|\.|\[|]|<|>|;|{|}|:|\'|"|\$|='
)

# for ignore_trivial_line()
# ignore fixed part
IGNORABLE_LINE_REGEX = re.compile(
    'Network Working Group|'
    'Request for Comments:|'
    'Obsoletes|'
    'Updates|'
    'Category|'
    'Table of Contents|'
    'Internet Engineering Task Force|'
    'ISSN|'
    'STD'
)
TOC_REGEX = re.compile(r'\.{5,}')

# for pre_remove()
# trivial characters such as separator
#     o - --> <-- === --- +-+ | ___ 000. 000.00
PRE_REMOVE_REGEX = re.compile(
    r'^o$|^-$|^-+>$|^<-+$|^=+$|^-+$|^[+-]+$|^\|$|^_+$|^(\d+\.)+\d*$'
)

# for post_remove()
# remove stop words and variables
ADDITIONAL_STOPWORDS = {
    'must', 'may', 'should', 'might', 'would', 'will',
    'i.e', 'optional', 'www.rfc-editor.org', 'see',
    'appendix', 'also', 'shall'
}
POST_REMOVE_REGEX = re.compile(
    r'^\d+$|'
    r'^(\d+\.)+\d*$|'
    r'^\d+-\d+$|'
    r'^rfc\d+$|'
    r'^[a-z](\.\d\d?)+$|'
    r'^0x[0-9abcdef]+$'
)


class RFCLoader:
    """RFC Loader from rfcyaml, with pickle-based cache.
    RFCLoader stores preprocessed RFC documents (i.e., segmented words)
    in the cache file.
    RFCLoader considers an RFC as a document."""

    default_filters = ["lower", "split_symbols", "lemmatize_verbs", "lemmatize_nns"]
    default_normalizer = Normalizer(default_filters)

    def __init__(self, cache_dir: str,
                 status_to_use: Optional[Iterable] = None,
                 normalizer: Optional[Normalizer] = None,
                 use_cache: bool = True):
        self._cache_dir = cache_dir
        self._status_to_use = status_to_use
        if normalizer is None:
            self._norm = self.default_normalizer
        else:
            self._norm = normalizer
        self._use_cache = use_cache

    @staticmethod
    def cache_name(rfcid) -> str:
        return "cache_rfc" + str(rfcid)

    def cache_path(self, rfcid) -> str:
        return os.path.join(self._cache_dir, self.cache_name(rfcid))

    def has_cache(self, rfcid) -> bool:
        return os.path.exists(self.cache_path(rfcid))

    def load(self, rfcid):
        with open(self.cache_path(rfcid), "rb", ) as f:
            return pickle.load(f)

    def dump(self, rfcid, obj):
        with open(self.cache_path(rfcid), "wb") as f:
            pickle.dump(obj, f)

    def preprocess(self, text):
        words = split_with_special_character(text)
        words = pre_remove(words)
        words = self._norm.process_line(words)
        tokens = post_remove(words)
        return tokens

    @staticmethod
    def _iter_components(rfc):
        text = ignore_trivial_line(rfc.lines)
        yield text

    def get_document(self, rfc):
        rfcid = rfc.n
        if self.has_cache(rfcid):
            if self._use_cache:
                return self.load(rfcid)

        documents = []
        for text in self._iter_components(rfc):
            documents.append(self.preprocess(text))
        if self._use_cache:
            self.dump(rfcid, documents)
        return documents

    def iter_all(self):
        if self._status_to_use is None:
            status_to_use = {
                RFCStatus.PROPOSED_STANDARD,
                RFCStatus.INTERNET_STANDARD,
                RFCStatus.DRAFT_STANDARD
            }
        else:
            status_to_use = self._status_to_use

        for rfcid in range(RFC_ID_MIN, RFC_ID_MAX):
            try:
                rfc = RFC(rfcid)
                if rfc.info.status in status_to_use:
                    yield rfc
            except FileNotFoundError:
                continue


class RFCSectionLoader(RFCLoader):
    """RFC Loader from rfcyaml, with pickle-based cache.
    RFCLoader stores preprocessed RFC documents (i.e., segmented words)
    in the cache file.
    RFCLoader considers a section as a document."""

    def __init__(self, cache_dir, status_to_use=None,
                 normalizer=None, use_cache=True):
        super().__init__(cache_dir, status_to_use,
                         normalizer, use_cache)

    @staticmethod
    def cache_name(rfcid) -> str:
        return "cache_rfc" + str(rfcid) + "_sections"

    @staticmethod
    def _iter_components(rfc):
        for section in rfc.sections:
            yield ignore_trivial_line(section.get_text())


class RFCLinesLoader(RFCLoader):
    """RFC Loader from rfcyaml, with pickle-based cache.
    RFCLoader stores preprocessed RFC documents (i.e., segmented words)
    in the cache file.
    RFCLoader considers a line as a document."""

    def __init__(self, cache_dir, status_to_use=None,
                 normalizer=None, use_cache=True):
        super().__init__(cache_dir, status_to_use,
                         normalizer, use_cache)

    @staticmethod
    def cache_name(rfcid) -> str:
        return "cache_rfc" + str(rfcid) + "_lines"

    @staticmethod
    def _iter_components(rfc):
        for line in rfc.lines:
            yield ignore_trivial_line([line])


def split_with_special_character(text: str) -> List[str]:
    """Split text with special characters."""
    removed_text = SEGMENTATION_REGEX.sub(' ', text)
    return removed_text.split()


def ignore_trivial_line(text: List[str]) -> str:
    """Ignore blank lines or fixed lines."""
    lines = []
    for line in text:
        if line.strip() == '':  # remove empty line
            continue
        if IGNORABLE_LINE_REGEX.match(line):  # remove header
            continue
        if TOC_REGEX.search(line):  # remove ToC
            continue
        lines.append(line)
    return ' '.join(lines)


def pre_remove(bow: List[str]) -> List[str]:
    """Remove trivial character in advance."""
    new_bow = []
    for word in bow:
        if word == '':
            continue
        if PRE_REMOVE_REGEX.match(word):
            continue
        new_bow.append(word)
    return new_bow


def post_remove(bow: List[str]) -> List[str]:
    """Remove stopwords."""
    new_bow = []
    stop_words = set(stopwords.words('english'))
    stop_words |= ADDITIONAL_STOPWORDS
    for word in bow:
        if word == '':
            continue
        if len(word) == 1:
            continue
        if word in stop_words:
            continue
        if POST_REMOVE_REGEX.match(word):
            continue
        else:
            new_bow.append(word)
    return new_bow


# def rfc_nlp_preprocess(rfc: RFC,
#                        normalizer: Optional[Normalizer] = None):
#     if normalizer is None:
#         normalizer = DEFAULT_NORMALIZER
#
#     text = ignore_trivial_line(rfc.lines)
#     words = split_with_special_character(text)
#     words = pre_remove(words)
#     words = normalizer.process_line(words)
#     tokens = post_remove(words)
#     return tokens
#
#
# def iter_all_docs(status_to_use=None):
#     if status_to_use is None:
#         status_to_use = {
#             RFCStatus.PROPOSED_STANDARD,
#             RFCStatus.INTERNET_STANDARD,
#             RFCStatus.DRAFT_STANDARD
#         }
#
#     for rfcid in range(RFC_ID_MIN, RFC_ID_MAX):
#         try:
#             rfc = RFC(rfcid)
#             if rfc.info.status in status_to_use:
#                 yield rfc
#         except FileNotFoundError:
#             continue

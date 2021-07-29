import re
from collections import Counter
from typing import Dict, List, Set

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from amulog import lt_regex


class Normalizer:
    _regex_strip_symbols = re.compile(r"^[^a-zA-Z0-9]*(?P<main>.*?)[^a-zA-Z0-9]*$")
    _regex_split_symbols = re.compile(r"[^a-zA-Z0-9]+")
    _regex_remove_symbols = re.compile(r"[^a-zA-Z0-9]+")

    _filter_manual_replace = "manual_replace"
    _filter_remove_variable = "remove_variable"

    def __init__(self, filters, mreplacer_sources, vreplacer_source,
                 host_alias_source=None, lemmatize_exception=None,
                 th_word_length=1):
        self._filters = filters
        self._th_word_length = th_word_length

        # init manual replacers
        n_manual_replacer = Counter(self._filters)[self._filter_manual_replace]
        if n_manual_replacer != len(mreplacer_sources):
            msg = "{0} manual_replacer sources should be given".format(
                n_manual_replacer)
            raise ValueError(msg)
        self._mreplacers = {}
        self._mreplacer_idx = {}
        rid = 0
        for idx, fil in enumerate(self._filters):
            if fil == self._filter_manual_replace:
                self._mreplacer_idx[idx] = rid
                rid += 1
        for rid, fp in enumerate(mreplacer_sources):
            self._mreplacers[rid] = self._parse_dict_file(fp)

        # init variable replacers
        if self._filter_remove_variable in self._filters:
            from amulog import host_alias
            if host_alias_source is None:
                ha = None
            else:
                ha = host_alias.HostAlias(host_alias_source)
            self._vreplacer = lt_regex.VariableRegex(vreplacer_source, ha)

        # init wordnet lemmatizer
        self._lemma = WordNetLemmatizer()
        if lemmatize_exception is None:
            lemmatize_exception = set()
        self._lemma_except = lemmatize_exception

    @staticmethod
    def _parse_dict_file(file_path):
        with open(file_path, 'r') as f:
            word_dict = {}
            for line in f:
                if ":" not in line:
                    continue
                key, words_str = [w.strip()
                                  for w in line.strip().split(":")]
                if words_str == "":
                    words = []
                else:
                    words = words_str.split()
                word_dict[key] = words
        return word_dict

    @staticmethod
    def manual_replace(sequence, replacer: Dict[str, List[str]]):
        new_sequence = []
        for w in sequence:
            if w in replacer:
                new_sequence += replacer[w]
            else:
                new_sequence.append(w)
        return new_sequence

    @staticmethod
    def remove_variable(sequence, replacer: lt_regex.VariableRegex):
        new_sequence = []
        for w in sequence:
            if not replacer.match(w):
                new_sequence.append(w)
        return new_sequence

    @staticmethod
    def remove_short_word(sequence: List[str], threshold):
        return [w for w in sequence if len(w) > threshold]

    @classmethod
    def _strip_word_symbol(cls, word):
        m = cls._regex_strip_symbols.match(word)
        if m:
            return m.group("main")
        else:
            # any message will match?
            return word

    @classmethod
    def strip_symbols(cls, sequence: List[str]):
        return [cls._strip_word_symbol(w) for w in sequence]

    def split_symbols(self, sequence: List[str]):
        new_sequence = []
        for w in sequence:
            new_sequence += self._regex_split_symbols.split(w)
        return new_sequence

    @classmethod
    def remove_symbols(cls, sequence: List[str]):
        return [cls._regex_remove_symbols.sub('', w)
                for w in sequence]

    @staticmethod
    def lower(sequence: List[str]):
        return [w.lower() for w in sequence]

    @staticmethod
    def lemmatize_verbs(sequence,
                        lemmatizer: WordNetLemmatizer,
                        exception: Set[str]):
        return [token if token in exception else
                lemmatizer.lemmatize(token, 'v')
                for token in sequence
                if token is not '']

    @staticmethod
    def lemmatize_nns(sequence,
                      lemmatizer: WordNetLemmatizer,
                      exception: Set[str]):
        return [token if token in exception else
                token if nltk.pos_tag([token])[0][1] == "NNS" else
                lemmatizer.lemmatize(token)
                for token in sequence
                if token is not '']

    def process_line(self, sequence, verbose=False):
        current_seq = sequence[:]
        for idx, fil in enumerate(self._filters):
            if verbose:
                print("before {0}: {1}".format(fil, current_seq))
            if fil == self._filter_manual_replace:
                current_seq = self.manual_replace(
                    current_seq, self._mreplacers[self._mreplacer_idx[idx]]
                )
            elif fil == self._filter_remove_variable:
                current_seq = self.remove_variable(
                    current_seq, self._vreplacer
                )
            elif fil == "remove_short_word":
                current_seq = self.remove_short_word(
                    current_seq, self._th_word_length
                )
            elif fil == "lemmatize_verbs":
                current_seq = self.lemmatize_verbs(
                    current_seq, self._lemma, self._lemma_except
                )
            elif fil == "lemmatize_nns":
                current_seq = self.lemmatize_nns(
                    current_seq, self._lemma, self._lemma_except
                )
            elif fil == "strip_symbols":
                current_seq = self.strip_symbols(current_seq)
            elif fil == "split_symbols":
                current_seq = self.split_symbols(current_seq)
            elif fil == "remove_symbols":
                current_seq = self.remove_symbols(current_seq)
            elif fil == "lower":
                current_seq = self.lower(current_seq)
            else:
                raise NotImplementedError
        if verbose:
            print("finally: {0}".format(current_seq))
        return current_seq

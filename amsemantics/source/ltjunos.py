import json
import re

from .base import Source
from amulog.lt_common import REPLACER_HEAD, REPLACER_TAIL, is_replacer

VARIABLE_REGEX = re.compile(
    r"^.*(?P<replace><variable>(?P<name>.*?)</variable>).*$"
)


class LTJunosSource(Source):

    def __init__(self, filepath, lognorm, use_replacer=False):
        self._filepath = filepath
        self._norm = lognorm
        self._use_replacer = use_replacer

        self._lp = self._init_ltjunos_parser()

    @staticmethod
    def _init_ltjunos_parser():
        from log2seq import preset, LogParser
        from log2seq.statement import StatementParser, Split, RemovePartial

        pattern_arrow = re.compile(r".*(?P<arrow>->).*")
        header_parser = preset.default_header_parsers()
        statement_rules = [
            RemovePartial([pattern_arrow], remove_groups=["arrow"], recursive=True),
            Split('"' + "/()[]{}|+',=><;`#: "),
        ]
        statement_parser = StatementParser(statement_rules)

        return LogParser(header_parser, statement_parser)

    @staticmethod
    def _iter_items(filepath):
        with open(filepath, "r") as f:
            obj = json.load(f)
            return obj["contents"]

    def _preprocess(self, text):
        words, _ = self._lp.process_statement(text)
        if self._use_replacer:
            # leave replacers as is, and preprocess other words
            preprocessed_words = []
            for word in words:
                if is_replacer(word):
                    preprocessed_words.append(word)
                else:
                    for new_word in self._norm.process_line([word]):
                        preprocessed_words.append(new_word)
        else:
            # preprocess all words except replacers
            words = [word for word in words
                     if not is_replacer(word)]
            preprocessed_words = self._norm.process_line(words)
        return preprocessed_words

    def load(self, **_):
        for item in self._iter_items(self._filepath):
            mes = "{0}: {1}".format(item["NAME"], item["MESSAGE"])
            while True:
                mo = VARIABLE_REGEX.match(mes)
                if mo:
                    attr_name = mo.group("name")
                    replacer = REPLACER_HEAD + attr_name.upper() + REPLACER_TAIL
                    attr_start, attr_end = mo.span("replace")
                    mes = mes[:attr_start] + replacer + mes[attr_end:]
                else:
                    break
            document = self._preprocess(mes)
            description = "LTJunos {0}tr A-Z a-z".format(mes)
            annotation = ("Junos", description)
            yield document, annotation

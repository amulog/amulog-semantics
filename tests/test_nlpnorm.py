#!/usr/bin/env python
# coding: utf-8

import os
import unittest

import log2seq
from amulog.alg.drain import drain
from amsemantics import Normalizer


TEST_REPLACER = "/".join((os.path.dirname(os.path.abspath(__file__)),
                         "../amsemantics/data/replacer.txt.sample"))
TEST_VARIABLE_REPLACER = drain.DEFAULT_REGEX_CONFIG


class TestNormalizer(unittest.TestCase):

    @staticmethod
    def _defaults():
        return {"filters": ["strip_symbols", "split_symbols",
                            "remove_variable", "remove_short_word",
                            "lower", "lemmatize_verbs", "lemmatize_nns",
                            "manual_replace"],
                "mreplacer_sources": [TEST_REPLACER],
                "vreplacer_source": TEST_VARIABLE_REPLACER,
                "lemmatize_exception": ["hoge"],
                "th_word_length": 1
                }

    def test_junos(self):
        tests = ["/usr/sbin/cron[**]: (root) CMD (newsyslog)",
                 ("rpd[**]: EVENT <Delete UpDown> ** index ** "
                  "<Broadcast Multicast> address #** **'"),
                 ("**:rpd[**]: RPD_OSPF_NBRDOWN: OSPF neighbor ** "
                  "(realm ** ** area **) state changed from ** to ** "
                  "due to KillNbr (event reason: interface went down)"),
                 "mib2d[34567]: lacp info not found for ifl:112"]
        answers = [['user', 'sbin', 'cron', 'root', 'command', 'newsyslog'],
                   ['rpd', 'event', 'delete', 'up', 'down', 'index',
                    'broadcast', 'multicast', 'address'],
                   ['rpd', 'rpd', 'ospf', 'neighbor', 'down', 'ospf',
                    'neighbor', 'realm', 'area', 'state', 'change', 'from',
                    'to', 'due', 'to', 'kill', 'neighbor', 'event', 'reason',
                    'interface', 'go', 'down'],
                   ['mib2d', 'lacp', 'information', 'not', 'find', 'for', 'ifl']]

        lp = log2seq.init_parser()
        ln = Normalizer(**self._defaults())
        for test, answer in zip(tests, answers):
            line = lp.process_statement(test)[0]
            result = ln.process_line(line, False)
            self.assertEqual(result, answer)

    def test_filter_remove_variable(self):
        tests = [["0", "1", "12345", "eth0", "10.5", "a"]]
        answers = [["eth0"]]

        ln = Normalizer(**self._defaults())
        for test, answer in zip(tests, answers):
            result = ln.process_line(test, False)
            self.assertEqual(result, answer)


if __name__ == "__main__":
    unittest.main()

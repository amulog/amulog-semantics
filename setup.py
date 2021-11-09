#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop


class Install(_install):
    def run(self):
        _install.run(self)
        import nltk
        nltk.download('wordnet')  # for nlpnorm
        nltk.download('averaged_perceptron_tagger')  # for nlpnorm
        nltk.download('stopwords')  # for source.rfcdoc


class Develop(_develop):
    def run(self):
        _develop.run(self)
        import nltk
        nltk.download('wordnet')  # for nlpnorm
        nltk.download('averaged_perceptron_tagger')  # for nlpnorm
        nltk.download('stopwords')  # for source.rfcdoc


def load_readme():
    with open('README.rst', 'r') as fd:
        return fd.read()


def load_requirements():
    """Parse requirements.txt"""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as fd:
        requirements = [line.rstrip() for line in fd]
    return requirements


package_name = 'amulog-semantics'
module_name = 'amsemantics'
data_dir = "/".join((module_name, "data"))

sys.path.append("./tests")
# data_files = ["/".join((data_dir, fn)) for fn in os.listdir(data_dir)]

init_path = os.path.join(os.path.dirname(__file__), module_name, '__init__.py')
with open(init_path) as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

setup(
    name=package_name,
    version=version,
    description='Semantic analysis extension of amulog',
    long_description=load_readme(),
    author='Satoru Kobayashi <sat@nii.ac.jp>, Kazuki Otomo <otomo@hongo.wide.ad.jp>',
    author_email='sat@nii.ac.jp',
    url='https://github.com/amulog/amulog-semantics/',
    install_requires=load_requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        "Intended Audience :: Developers",
        'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    cmdclass={"develop": Develop,
              "install": Install},
    license='The 3-Clause BSD License',
  
    packages=find_packages(),
#    package_data={'amsemantics': data_files},
    include_package_data=True,
)

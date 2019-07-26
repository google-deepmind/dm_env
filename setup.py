# pylint: disable=g-bad-file-header
# Copyright 2019 The dm_env Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Install script for setuptools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
from setuptools import find_packages
from setuptools import setup

setup(
    name='dm_env',
    version=imp.load_source('_metadata', 'dm_env/_metadata.py').__version__,
    description='A Python interface for Reinforcement Learning environments.',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'absl-py',
        'enum34',
        'funcsigs ; python_version < "3.3"',
        'numpy',
        'six',
    ],
    tests_require=[
        'nose',
    ],
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

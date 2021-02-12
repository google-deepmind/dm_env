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
"""A Python interface for reinforcement learning environments."""

from dm_env import _environment
from dm_env._metadata import __version__

Environment = _environment.Environment
StepType = _environment.StepType
TimeStep = _environment.TimeStep

# Helper functions for creating TimeStep namedtuples with default settings.
restart = _environment.restart
termination = _environment.termination
transition = _environment.transition
truncation = _environment.truncation

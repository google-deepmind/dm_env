# pylint: disable=g-bad-file-header
# Copyright 2021 The dm_env Authors. All Rights Reserved.
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
"""Tests for auto_reset_environment."""

from absl.testing import absltest
from dm_env import _environment
from dm_env import auto_reset_environment
from dm_env import specs
from dm_env import test_utils
import numpy as np


class FakeEnvironment(auto_reset_environment.AutoResetEnvironment):
  """Environment that resets after a given number of steps."""

  def __init__(self, step_limit):
    super(FakeEnvironment, self).__init__()
    self._step_limit = step_limit
    self._steps_taken = 0

  def _reset(self):
    self._steps_taken = 0
    return _environment.restart(observation=np.zeros(3))

  def _step(self, action):
    self._steps_taken += 1
    if self._steps_taken < self._step_limit:
      return _environment.transition(reward=0.0, observation=np.zeros(3))
    else:
      return _environment.termination(reward=0.0, observation=np.zeros(3))

  def action_spec(self):
    return specs.Array(shape=(), dtype='int')

  def observation_spec(self):
    return specs.Array(shape=(3,), dtype='float')


class AutoResetEnvironmentTest(test_utils.EnvironmentTestMixin,
                               absltest.TestCase):

  def make_object_under_test(self):
    return FakeEnvironment(step_limit=5)

  def make_action_sequence(self):
    for _ in range(20):
      yield np.array(0)


if __name__ == '__main__':
  absltest.main()

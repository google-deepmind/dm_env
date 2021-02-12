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
"""Tests for dm_env.test_utils."""

import itertools

from absl.testing import absltest
import dm_env
from dm_env import specs
from dm_env import test_utils
import numpy as np

REWARD_SPEC = specs.Array(shape=(), dtype=float)
DISCOUNT_SPEC = specs.BoundedArray(shape=(), dtype=float, minimum=0, maximum=1)
OBSERVATION_SPEC = specs.Array(shape=(2, 3), dtype=float)
ACTION_SPEC = specs.BoundedArray(shape=(), dtype=int, minimum=0, maximum=2)

REWARD = REWARD_SPEC.generate_value()
DISCOUNT = DISCOUNT_SPEC.generate_value()
OBSERVATION = OBSERVATION_SPEC.generate_value()

FIRST = dm_env.restart(observation=OBSERVATION)
MID = dm_env.transition(
    reward=REWARD, observation=OBSERVATION, discount=DISCOUNT)
LAST = dm_env.truncation(
    reward=REWARD, observation=OBSERVATION, discount=DISCOUNT)


class MockEnvironment(dm_env.Environment):

  def __init__(self, timesteps):
    self._timesteps = timesteps
    self._iter_timesteps = itertools.cycle(self._timesteps)

  def reset(self):
    self._iter_timesteps = itertools.cycle(self._timesteps)
    return next(self._iter_timesteps)

  def step(self, action):
    return next(self._iter_timesteps)

  def reward_spec(self):
    return REWARD_SPEC

  def discount_spec(self):
    return DISCOUNT_SPEC

  def action_spec(self):
    return ACTION_SPEC

  def observation_spec(self):
    return OBSERVATION_SPEC


def _make_test_case_with_expected_failures(
    name,
    timestep_sequence,
    expected_failures):

  class NewTestCase(test_utils.EnvironmentTestMixin, absltest.TestCase):

    def make_object_under_test(self):
      return MockEnvironment(timestep_sequence)

  for method_name, exception_type in expected_failures:
    def wrapped_method(
        self, method_name=method_name, exception_type=exception_type):
      super_method = getattr(super(NewTestCase, self), method_name)
      with self.assertRaises(exception_type):
        return super_method()
    setattr(NewTestCase, method_name, wrapped_method)

  NewTestCase.__name__ = name
  return NewTestCase


TestValidTimestepSequence = _make_test_case_with_expected_failures(
    name='TestValidTimestepSequence',
    timestep_sequence=[FIRST, MID, MID, LAST],
    expected_failures=[],
)


# Sequences where the ordering of StepTypes is invalid.


TestTwoFirstStepsInARow = _make_test_case_with_expected_failures(
    name='TestTwoFirstStepsInARow',
    timestep_sequence=[FIRST, FIRST, MID, MID, LAST],
    expected_failures=[
        ('test_longer_action_sequence', AssertionError),
        ('test_step_after_reset', AssertionError),
        ('test_step_on_fresh_environment', AssertionError),
    ],
)


TestStartsWithMid = _make_test_case_with_expected_failures(
    name='TestStartsWithMid',
    timestep_sequence=[MID, MID, LAST],
    expected_failures=[
        ('test_longer_action_sequence', AssertionError),
        ('test_reset', AssertionError),
        ('test_step_after_reset', AssertionError),
        ('test_step_on_fresh_environment', AssertionError),
    ],
)

TestMidAfterLast = _make_test_case_with_expected_failures(
    name='TestMidAfterLast',
    timestep_sequence=[FIRST, MID, LAST, MID],
    expected_failures=[
        ('test_longer_action_sequence', AssertionError),
    ],
)

TestFirstAfterMid = _make_test_case_with_expected_failures(
    name='TestFirstAfterMid',
    timestep_sequence=[FIRST, MID, FIRST],
    expected_failures=[
        ('test_longer_action_sequence', AssertionError),
    ],
)

# Sequences where one or more TimeSteps have invalid contents.


TestFirstStepHasReward = _make_test_case_with_expected_failures(
    name='TestFirstStepHasReward',
    timestep_sequence=[
        FIRST._replace(reward=1.0),  # Should be None.
        MID,
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_reset', AssertionError),
        ('test_step_after_reset', AssertionError),
        ('test_step_on_fresh_environment', AssertionError),
        ('test_longer_action_sequence', AssertionError),
    ]
)

TestFirstStepHasDiscount = _make_test_case_with_expected_failures(
    name='TestFirstStepHasDiscount',
    timestep_sequence=[
        FIRST._replace(discount=1.0),  # Should be None.
        MID,
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_reset', AssertionError),
        ('test_step_after_reset', AssertionError),
        ('test_step_on_fresh_environment', AssertionError),
        ('test_longer_action_sequence', AssertionError),
    ]
)

TestInvalidReward = _make_test_case_with_expected_failures(
    name='TestInvalidReward',
    timestep_sequence=[
        FIRST,
        MID._replace(reward=False),  # Should be a float.
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_step_after_reset', ValueError),
        ('test_step_on_fresh_environment', ValueError),
        ('test_longer_action_sequence', ValueError),
    ]
)

TestInvalidDiscount = _make_test_case_with_expected_failures(
    name='TestInvalidDiscount',
    timestep_sequence=[
        FIRST,
        MID._replace(discount=1.5),  # Should be between 0 and 1.
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_step_after_reset', ValueError),
        ('test_step_on_fresh_environment', ValueError),
        ('test_longer_action_sequence', ValueError),
    ]
)

TestInvalidObservation = _make_test_case_with_expected_failures(
    name='TestInvalidObservation',
    timestep_sequence=[
        FIRST,
        MID._replace(observation=np.zeros((3, 4))),  # Wrong shape.
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_step_after_reset', ValueError),
        ('test_step_on_fresh_environment', ValueError),
        ('test_longer_action_sequence', ValueError),
    ]
)

TestMismatchingObservationStructure = _make_test_case_with_expected_failures(
    name='TestInvalidObservation',
    timestep_sequence=[
        FIRST,
        MID._replace(observation=[OBSERVATION]),  # Wrong structure.
        MID,
        LAST,
    ],
    expected_failures=[
        ('test_step_after_reset', AssertionError),
        ('test_step_on_fresh_environment', AssertionError),
        ('test_longer_action_sequence', AssertionError),
    ]
)


if __name__ == '__main__':
  absltest.main()

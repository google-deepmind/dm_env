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
"""Tests for dm_env._environment."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env


class TimeStepHelpersTest(parameterized.TestCase):

  @parameterized.parameters(dict(observation=-1), dict(observation=[2., 3.]))
  def test_restart(self, observation):
    time_step = dm_env.restart(observation)
    self.assertIs(dm_env.StepType.FIRST, time_step.step_type)
    self.assertEqual(observation, time_step.observation)
    self.assertIsNone(time_step.reward)
    self.assertIsNone(time_step.discount)

  @parameterized.parameters(
      dict(observation=-1., reward=2.0, discount=1.0),
      dict(observation=(2., 3.), reward=0., discount=0.))
  def test_transition(self, observation, reward, discount):
    time_step = dm_env.transition(
        reward=reward, observation=observation, discount=discount)
    self.assertIs(dm_env.StepType.MID, time_step.step_type)
    self.assertEqual(observation, time_step.observation)
    self.assertEqual(reward, time_step.reward)
    self.assertEqual(discount, time_step.discount)

  @parameterized.parameters(
      dict(observation=-1., reward=2.0),
      dict(observation=(2., 3.), reward=0.))
  def test_termination(self, observation, reward):
    time_step = dm_env.termination(reward=reward, observation=observation)
    self.assertIs(dm_env.StepType.LAST, time_step.step_type)
    self.assertEqual(observation, time_step.observation)
    self.assertEqual(reward, time_step.reward)
    self.assertEqual(0.0, time_step.discount)

  @parameterized.parameters(
      dict(observation=-1., reward=2.0, discount=1.0),
      dict(observation=(2., 3.), reward=0., discount=0.))
  def test_truncation(self, reward, observation, discount):
    time_step = dm_env.truncation(reward, observation, discount)
    self.assertIs(dm_env.StepType.LAST, time_step.step_type)
    self.assertEqual(observation, time_step.observation)
    self.assertEqual(reward, time_step.reward)
    self.assertEqual(discount, time_step.discount)

  @parameterized.parameters(
      dict(step_type=dm_env.StepType.FIRST,
           is_first=True, is_mid=False, is_last=False),
      dict(step_type=dm_env.StepType.MID,
           is_first=False, is_mid=True, is_last=False),
      dict(step_type=dm_env.StepType.LAST,
           is_first=False, is_mid=False, is_last=True),
  )
  def test_step_type_helpers(self, step_type, is_first, is_mid, is_last):
    time_step = dm_env.TimeStep(
        reward=None, discount=None, observation=None, step_type=step_type)

    with self.subTest('TimeStep methods'):
      self.assertEqual(is_first, time_step.first())
      self.assertEqual(is_mid, time_step.mid())
      self.assertEqual(is_last, time_step.last())

    with self.subTest('StepType methods'):
      self.assertEqual(is_first, time_step.step_type.first())
      self.assertEqual(is_mid, time_step.step_type.mid())
      self.assertEqual(is_last, time_step.step_type.last())


if __name__ == '__main__':
  absltest.main()

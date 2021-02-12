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
"""Reusable fixtures for testing implementations of `dm_env.Environment`.

This generally kicks the tyres on an environment, and checks that it complies
with the interface contract for `dm_env.Environment`.

To test your own environment, all that's required is to inherit from
`EnvironmentTestMixin` and `absltest.TestCase` (in this order), overriding
`make_object_under_test`:

```python
from absl.testing import absltest
from dm_env import test_utils

class MyEnvImplementationTest(test_utils.EnvironmentTestMixin,
                              absltest.TestCase):

  def make_object_under_test(self):
    return my_env.MyEnvImplementation()
```

We recommend that you also override `make_action_sequence` in order to generate
a sequence of actions that covers any 'interesting' behaviour in your
environment. For episodic environments in particular, we recommend returning an
action sequence that allows the environment to reach the end of an episode,
otherwise the contract around end-of-episode behaviour will not be checked. The
default implementation of `make_action_sequence` simply generates a dummy action
conforming to the `action_spec` and repeats it 20 times.

You can also add your own tests alongside the defaults if you want to test some
behaviour that's specific to your environment. There are some assertions and
helpers here which may be useful to you in writing these tests.

Note that we disable the pytype: attribute-error static check for the mixin as
absltest.TestCase methods aren't statically available here, only once mixed in.
"""

from absl import logging
import dm_env
import tree
from dm_env import _abstract_test_mixin
_STEP_NEW_ENV_MUST_RETURN_FIRST = (
    "calling step() on a fresh environment must produce a step with "
    "step_type FIRST, got {}")
_RESET_MUST_RETURN_FIRST = (
    "reset() must produce a step with step_type FIRST, got {}.")
_FIRST_MUST_NOT_HAVE_REWARD = "a FIRST step must not have a reward."
_FIRST_MUST_NOT_HAVE_DISCOUNT = "a FIRST step must not have a discount."
_STEP_AFTER_FIRST_MUST_NOT_RETURN_FIRST = (
    "calling step() after a FIRST step must not produce another FIRST.")
_FIRST_MUST_COME_AFTER_LAST = (
    "step() must produce a FIRST step after a LAST step.")
_FIRST_MUST_ONLY_COME_AFTER_LAST = (
    "step() must only produce a FIRST step after a LAST step "
    "or on a fresh environment.")


class EnvironmentTestMixin(_abstract_test_mixin.TestMixin):
  """Mixin to help test implementations of `dm_env.Environment`.

  Subclasses must override `make_object_under_test` to return an instance of the
  `Environment` to be tested.
  """

  @property
  def environment(self):
    """An alias of `self.object_under_test`, for readability."""
    return self.object_under_test

  def tearDown(self):
    self.environment.close()
    # A call to super is required for cooperative multiple inheritance to work.
    super().tearDown()  # pytype: disable=attribute-error

  def make_action_sequence(self):
    """Generates a sequence of actions for a longer test.

    Yields:
      A sequence of actions compatible with environment's action_spec().

    Ideally you should override this to generate an action sequence that will
    trigger an end of episode, in order to ensure this behaviour is tested.
    Otherwise it will just repeat a test value conforming to the action spec
    20 times.
    """
    for _ in range(20):
      yield self.make_action()

  def make_action(self):
    """Returns a single action conforming to the environment's action_spec()."""
    spec = self.environment.action_spec()
    return tree.map_structure(lambda s: s.generate_value(), spec)

  def reset_environment(self):
    """Resets the environment and checks that the returned TimeStep is valid.

    Returns:
      The TimeStep instance returned by reset().
    """
    step = self.environment.reset()
    self.assertValidStep(step)
    self.assertIs(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                  _RESET_MUST_RETURN_FIRST.format(step.step_type))
    return step

  def step_environment(self, action=None):
    """Steps the environment and checks that the returned TimeStep is valid.

    Args:
      action: Optional action conforming to the environment's action_spec(). If
        None then a valid action will be generated.

    Returns:
      The TimeStep instance returned by step(action).
    """
    if action is None:
      action = self.make_action()
    step = self.environment.step(action)
    self.assertValidStep(step)
    return step

  # Custom assertions
  # ----------------------------------------------------------------------------

  def assertValidStep(self, step):
    """Checks that a TimeStep conforms to the environment's specs.

    Args:
      step: An instance of TimeStep.
    """
    # pytype: disable=attribute-error
    self.assertIsInstance(step, dm_env.TimeStep)
    self.assertIsInstance(step.step_type, dm_env.StepType)
    if step.step_type is dm_env.StepType.FIRST:
      self.assertIsNone(step.reward, _FIRST_MUST_NOT_HAVE_REWARD)
      self.assertIsNone(step.discount, _FIRST_MUST_NOT_HAVE_DISCOUNT)
    else:
      self.assertValidReward(step.reward)
      self.assertValidDiscount(step.discount)
    self.assertValidObservation(step.observation)
    # pytype: enable=attribute-error

  def assertConformsToSpec(self, value, spec):
    """Checks that `value` conforms to `spec`.

    Args:
      value: A potentially nested structure of numpy arrays or scalars.
      spec: A potentially nested structure of `specs.Array` instances.
    """
    try:
      tree.assert_same_structure(value, spec)
    except (TypeError, ValueError) as e:
      self.fail("`spec` and `value` have mismatching structures: {}".format(e))  # pytype: disable=attribute-error
    def validate(path, item, array_spec):
      try:
        return array_spec.validate(item)
      except ValueError as e:
        raise ValueError("Value at path {!r} failed validation: {}."
                         .format("/".join(map(str, path)), e))
    tree.map_structure_with_path(validate, value, spec)

  def assertValidObservation(self, observation):
    """Checks that `observation` conforms to the `observation_spec()`."""
    self.assertConformsToSpec(observation, self.environment.observation_spec())

  def assertValidReward(self, reward):
    """Checks that `reward` conforms to the `reward_spec()`."""
    self.assertConformsToSpec(reward, self.environment.reward_spec())

  def assertValidDiscount(self, discount):
    """Checks that `discount` conforms to the `discount_spec()`."""
    self.assertConformsToSpec(discount, self.environment.discount_spec())

  # Test cases
  # ----------------------------------------------------------------------------

  def test_reset(self):
    # Won't hurt to check this works twice in a row:
    for _ in range(2):
      self.reset_environment()

  def test_step_on_fresh_environment(self):
    # Calling `step()` on a fresh environment should be equivalent to `reset()`.
    # Note that the action should be ignored.
    step = self.step_environment()
    self.assertIs(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                  _STEP_NEW_ENV_MUST_RETURN_FIRST.format(step.step_type))
    step = self.step_environment()
    self.assertIsNot(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                     _STEP_AFTER_FIRST_MUST_NOT_RETURN_FIRST)

  def test_step_after_reset(self):
    for _ in range(2):
      self.reset_environment()
      step = self.step_environment()
      self.assertIsNot(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                       _STEP_AFTER_FIRST_MUST_NOT_RETURN_FIRST)

  def test_longer_action_sequence(self):
    """Steps the environment using actions generated by `make_action_sequence`.

    The sequence of TimeSteps returned are checked for validity.
    """
    encountered_last_step = False
    for _ in range(2):
      step = self.reset_environment()
      prev_step_type = step.step_type
      for action in self.make_action_sequence():
        step = self.step_environment(action)
        if prev_step_type is dm_env.StepType.LAST:
          self.assertIs(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                        _FIRST_MUST_COME_AFTER_LAST)
        else:
          self.assertIsNot(dm_env.StepType.FIRST, step.step_type,  # pytype: disable=attribute-error
                           _FIRST_MUST_ONLY_COME_AFTER_LAST)
        if step.last():
          encountered_last_step = True
        prev_step_type = step.step_type
    if not encountered_last_step:
      logging.info(
          "Could not test the contract around end-of-episode behaviour. "
          "Consider implementing `make_action_sequence` so that an end of "
          "episode is reached.")
    else:
      logging.info("Successfully checked end of episode.")

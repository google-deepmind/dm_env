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
"""Python RL Environment API."""

import abc
import enum
from typing import Any, NamedTuple

from dm_env import specs


class TimeStep(NamedTuple):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    step_type: A `StepType` enum value.
    reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
      sequence.
    discount: A scalar, NumPy array, nested dict, list or tuple of discount
      values in the range `[0, 1]`, or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
      Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
      also valid in place of a scalar array.
  """

  # TODO(b/143116886): Use generics here when PyType supports them.
  step_type: Any
  reward: Any
  discount: Any
  observation: Any

  def first(self) -> bool:
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == StepType.MID

  def last(self) -> bool:
    return self.step_type == StepType.LAST


class StepType(enum.IntEnum):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = 0
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = 1
  # Denotes the last `TimeStep` in a sequence.
  LAST = 2

  def first(self) -> bool:
    return self is StepType.FIRST

  def mid(self) -> bool:
    return self is StepType.MID

  def last(self) -> bool:
    return self is StepType.LAST


class Environment(metaclass=abc.ABCMeta):
  """Abstract base class for Python RL environments.

  Observations and valid actions are described with `Array` specs, defined in
  the `specs` module.
  """

  @abc.abstractmethod
  def reset(self) -> TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: `None`, indicating the reward is undefined.
        discount: `None`, indicating the discount is undefined.
        observation: A NumPy array, or a nested dict, list or tuple of arrays.
          Scalar values that can be cast to NumPy arrays (e.g. Python floats)
          are also valid in place of a scalar array. Must conform to the
          specification returned by `observation_spec()`.
    """

  @abc.abstractmethod
  def step(self, action) -> TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `action`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `action` will be ignored.

    Args:
      action: A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep, or None if step_type is
          `StepType.FIRST`. Must conform to the specification returned by
          `reward_spec()`.
        discount: A discount in the range [0, 1], or None if step_type is
          `StepType.FIRST`. Must conform to the specification returned by
          `discount_spec()`.
        observation: A NumPy array, or a nested dict, list or tuple of arrays.
          Scalar values that can be cast to NumPy arrays (e.g. Python floats)
          are also valid in place of a scalar array. Must conform to the
          specification returned by `observation_spec()`.
    """

  def reward_spec(self):
    """Describes the reward returned by the environment.

    By default this is assumed to be a single float.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.Array(shape=(), dtype=float, name='reward')

  def discount_spec(self):
    """Describes the discount returned by the environment.

    By default this is assumed to be a single float between 0 and 1.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.BoundedArray(
        shape=(), dtype=float, minimum=0., maximum=1., name='discount')

  @abc.abstractmethod
  def observation_spec(self):
    """Defines the observations provided by the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """

  @abc.abstractmethod
  def action_spec(self):
    """Defines the actions that should be provided to `step`.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """

  def close(self):
    """Frees any resources used by the environment.

    Implement this method for an environment backed by an external process.

    This method can be used directly

    ```python
    env = Env(...)
    # Use env.
    env.close()
    ```

    or via a context manager

    ```python
    with Env(...) as env:
      # Use env.
    ```
    """
    pass

  def __enter__(self):
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Allows the environment to be used in a with-statement context."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()


# Helper functions for creating TimeStep namedtuples with default settings.


def restart(observation):
  """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`."""
  return TimeStep(StepType.FIRST, None, None, observation)


def transition(reward, observation, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.MID`."""
  return TimeStep(StepType.MID, reward, discount, observation)


def termination(reward, observation):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
  return TimeStep(StepType.LAST, reward, 0.0, observation)


def truncation(reward, observation, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
  return TimeStep(StepType.LAST, reward, discount, observation)

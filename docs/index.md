# Environment API and Semantics

This text describes the Python-based Environment API defined by `dm_env`.

## Overview

The main interaction with an environment is via the `step()` method.

Each call to an environment's `step()` method takes an `action` parameter
and returns a `TimeStep` namedtuple with fields

```none
step_type, reward, discount, observation
```

A **sequence** consists of a series of `TimeStep`s returned by consecutive calls
to `step()`. In many settings we refer to each sequence as an *episode*. Each
sequence starts with a `step_type` of `FIRST`, ends with a `step_type` of
`LAST`, and has a `step_type` of `MID` for all intermediate `TimeStep`s.

As well as `step()`, each environment implements a `reset()` method. This takes
no arguments, forces the start of a new sequence and returns the first
`TimeStep`. See the [run loop samples](#run-loop-samples) below for more
details.

Calling `step()` on a new environment instance, or immediately after a
`TimeStep` with a `step_type` of `LAST` is equivalent to calling `reset()`. In
other words, the `action` argument will be ignored and a new sequence will
begin, starting with a `step_type` of `FIRST`.

NOTE: The `discount` does *not* determine when a sequence ends. The `discount`
may be 0 in the middle of a sequence and ≥0 at the end of a sequence.

### Example sequences

We show two examples of sequences below, along with the first `TimeStep` of the
next sequence.

Each row corresponds to the tuple returned by an environment's `step()` method.
We use `r`, `ɣ` and `obs` to denote the reward, discount and observation
respectively, `x` to denote a `None` or optional value at a timestep, and `✓`
to denote a value that exists at a timestep.

Example: A sequence where the end of the *prediction*—the discounted sum of
future rewards that we wish to predict—coincides with the end of the sequence.
i.e., this sequence ends with a discount of 0. Such a sequence could represent a
single episode of a *finite-horizon* RL task.

```none
(r, ɣ, obs)  | (x, x, ✓) →  (✓, ✓, ✓)  →  (✓, 0, ✓) ⇢ (x, x, ✓)
step_type    |   FIRST         MID           LAST       FIRST
```

Example: Here the prediction does not terminate at the end of the sequence,
which ends with a nonzero discount. This type of termination is sometimes used
in *infinite-horizon* RL settings.

```none
(r, ɣ, obs)  | (x, x, ✓) →  (✓, ✓, ✓)  →  (✓, > 0, ✓) ⇢ (x, x, ✓)
step_type    |   FIRST         MID           LAST         FIRST
```

In general, a discount of `0` does not need to coincide with the end of a
sequence. An environment may return `ɣ = 0` in the middle of a sequence, and
may do this multiple times within a sequence. We do not (typically) call these
sub-sequences episodes.

The `step_type` can potentially be used by an agent. For instance, some agents
may reset their short-term memory when `step_type` is `LAST`, but not when the
`step_type` is `MID`, even if the discount is `0`. This is up to the
creator of the agent, but it does mean that the aforementioned two ways to
model a termination of the prediction do not necessarily correspond to the same
agent behaviour.

## Run loop samples

Here we show some sample run loops for using an environment with an agent class
that implements a `step(timestep)` method.

NOTE: Environments do not make any assumptions about the structure of
algorithmic code or agent classes. These examples are illustrative only.

### Continuing

We may call `step()` repeatedly.

```python
timestep = env.reset()
while True:
  action = agent.step(timestep)
  timestep = env.step(action)

```
NOTE: An environment will ignore `action` after a `LAST` step, and return the
`FIRST` step of a new sequence. An agent or algorithm may use the `step_type`,
for example to decide when to reset short-term memory.

### Set number of sequences

We can choose to run a specific number of sequences. Here we use the syntactic
sugar method `.last()` to check whether we are at the end of a sequence.

```python
for _ in range(num_sequences):

  timestep = env.reset()
  while True:
    action = agent.step(timestep)
    timestep = env.step(action)
    if timestep.last():
      _ = agent.step(timestep)
      break
```

A `TimeStep` also has `.first()` and `.mid()` methods.

### Manual truncation

We can truncate a sequence manually at some `step_limit`.

```python
step_limit = 100
for _ in range(num_sequences):

  timestep = env.reset()

  step_counter = 1
  while True:
    action = agent.step(timestep)
    timestep = env.step(action)
    if step_counter == step_limit:
      timestep = timestep._replace(step_type=environment.StepType.LAST)

    if timestep.last():
      _ = agent.step(timestep)
      break

    step_counter += 1
```

In this example we've accessed the `step_type` element directly.

## The format of observations and actions

Environments should return observations and accept actions in the form of
[NumPy arrays][numpy_array].

An environment may return observations made up of multiple arrays, for example a
list where the first item is an array containing an RGB image and the second
item is an array containing velocities. The arrays may also be values in a
`dict`, or any other structure made up of basic Python containers. Note: A
single array is a perfectly valid format.

Similarly, actions may be specified as multiple arrays, for example control
signals for distinct parts of a simulated robot.

Each environment also implements an `observation_spec()` and an `action_spec()`
method. Each method should return a structure of [`Array` specs][specs],
where the structure should correspond exactly to the
format of the actions/observations.

Each `Array` spec should define the `dtype`, `shape` and, where possible, the
bounds and name of the corresponding action or observation array.

Note: Actions should almost always specify bounds, e.g. they should use the
[`BoundedArray` spec][specs] subclass.

[numpy_array]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
[specs]: ../dm_env/specs.py

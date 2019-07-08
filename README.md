# `dm_env`: The DeepMind RL Environment API

This package describes an interface for Python reinforcement learning (RL)
environments. It consists of the following core components:

*   `dm_env.Environment`: An abstract base class for RL environments.
*   `dm_env.TimeStep`: A container class representing the outputs of the
    environment on each time step (transition).
*   `dm_env.specs`: A module containing primitives that are used to describe the
    format of the actions consumed by an environment, as well as the
    observations, rewards, and discounts it returns.

Please see the documentation [here][api_docs] for more information about the
semantics of the environment interface and how to use it.

## Installation

`dm_env` can be installed directly from our GitHub repository using `pip`:

```bash
pip install git+git://github.com/deepmind/dm_env.git
```

or alternatively by checking out a local copy of our repository and running:

```bash
pip install /path/to/local/dm_env/
```

[api_docs]: docs/index.md

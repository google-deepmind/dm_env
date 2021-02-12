# `dm_env`: The DeepMind RL Environment API

This package describes an interface for Python reinforcement learning (RL)
environments. It consists of the following core components:

*   `dm_env.Environment`: An abstract base class for RL environments.
*   `dm_env.TimeStep`: A container class representing the outputs of the
    environment on each time step (transition).
*   `dm_env.specs`: A module containing primitives that are used to describe the
    format of the actions consumed by an environment, as well as the
    observations, rewards, and discounts it returns.
*   `dm_env.test_utils`: Tools for testing whether concrete environment
    implementations conform to the `dm_env.Environment` interface.

Please see the documentation [here][api_docs] for more information about the
semantics of the environment interface and how to use it. The [examples]
subdirectory also contains illustrative examples of RL environments implemented
using the `dm_env` interface.

## Installation

`dm_env` can be installed from PyPI using `pip`:

```bash
pip install dm-env
```

Note that from version 1.4 onwards, we support Python 3.6+ only.

You can also install it directly from our GitHub repository using `pip`:

```bash
pip install git+git://github.com/deepmind/dm_env.git
```

or alternatively by checking out a local copy of our repository and running:

```bash
pip install /path/to/local/dm_env/
```

[api_docs]: docs/index.md
[examples]: examples/

## Citing

To cite this repository:

```bibtex
@misc{dm_env2019,
  author={Alistair Muldal and
      Yotam Doron and
      John Aslanides and
      Tim Harley and
      Tom Ward and
      Siqi Liu},
  title={dm\_env: A Python interface for reinforcement learning environments},
  year={2019},
  url={http://github.com/deepmind/dm_env}
}
```

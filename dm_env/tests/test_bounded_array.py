import numpy as np
from dm_env import specs

def test_generate_value_within_bounds():
    spec = specs.BoundedArray(
        shape=(2, 2),
        dtype=np.float32,
        minimum=0.0,
        maximum=5.0
    )
    value = spec.generate_value()

    assert value.shape == (2, 2)
    assert value.dtype == np.float32
    assert np.all(value >= 0.0)
    assert np.all(value <= 5.0)

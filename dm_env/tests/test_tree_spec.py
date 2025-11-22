from dataclasses import dataclass
from dm_env import specs

@dataclass
class MyAction:
    a: specs.Array
    b: specs.Array

def test_tree_spec_generate_value():
    spec = specs.TreeSpec(
        MyAction(
            a=specs.Array(shape=(2,), dtype=float),
            b=specs.Array(shape=(3,), dtype=float)
        )
    )

    value = spec.generate_value()

    # type check
    assert isinstance(value, MyAction)

    # shape checks
    assert value.a.shape == (2,)
    assert value.b.shape == (3,)

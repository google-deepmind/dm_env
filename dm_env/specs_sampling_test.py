# Copyright 2026 The dm_env Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for dtype- and bounds-aware random spec sampling."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np


class SamplingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int8', np.int8),
      ('uint8', np.uint8),
      ('int64', np.int64),
      ('uint64', np.uint64),
      ('float16', np.float16),
      ('float32', np.float32),
      ('float64', np.float64),
      ('complex64', np.complex64),
      ('complex128', np.complex128),
      ('bool', np.bool_),
  )
  def test_array_samples_match_dtype_and_shape(self, dtype):
    spec = specs.Array((4, 7), dtype)
    result = spec.sample(np.random.default_rng(17))
    self.assertEqual(result.dtype, spec.dtype)
    self.assertEqual(result.shape, (4, 7))
    np.testing.assert_array_equal(spec.validate(result), result)

  def test_array_scalar_and_empty_shapes(self):
    for shape in ((), (0,), (2, 0, 3)):
      with self.subTest(shape=shape):
        spec = specs.Array(shape, np.float32)
        sample = spec.sample()
        self.assertEqual(sample.shape, shape)
        spec.validate(sample)

  def test_array_reproducible_with_seeded_generator(self):
    spec = specs.Array((128,), np.float32)
    first = spec.sample(np.random.default_rng(123))
    second = spec.sample(np.random.default_rng(123))
    np.testing.assert_array_equal(first, second)
    self.assertGreater(np.unique(first).size, 50)

  def test_unbounded_integer_sampling_can_use_full_uint64_dtype(self):
    spec = specs.Array((20,), np.uint64)
    values = spec.sample(np.random.default_rng(23))
    spec.validate(values)
    self.assertGreater(values.max(), np.iinfo(np.int64).max)

  def test_unbounded_object_dtype_requires_specialized_spec(self):
    spec = specs.Array((2,), object)
    with self.assertRaisesRegex(TypeError, 'non-numeric dtype'):
      spec.sample(np.random.default_rng(1))

  def test_bounded_integer_samples_respect_broadcast_bounds(self):
    spec = specs.BoundedArray(
        shape=(400, 3),
        dtype=np.int16,
        minimum=[-10, 3, 7],
        maximum=[12, 3, 8],
    )
    sample = spec.sample(np.random.default_rng(7))
    spec.validate(sample)
    self.assertTrue(np.all(sample[:, 1] == 3))
    self.assertEqual(set(np.unique(sample[:, 2])), {7, 8})
    self.assertTrue(np.any(sample[:, 0] == -10))
    self.assertTrue(np.any(sample[:, 0] == 12))

  def test_bounded_unsigned_integer_including_uint64_maximum(self):
    max_value = np.iinfo(np.uint64).max
    spec = specs.BoundedArray(
        (40,), np.uint64, minimum=max_value - 1, maximum=max_value
    )
    sample = spec.sample(np.random.default_rng(44))
    spec.validate(sample)
    self.assertEqual(set(sample.tolist()), {max_value - 1, max_value})

  def test_bounded_float_samples_match_heterogeneous_bounds(self):
    spec = specs.BoundedArray(
        (500, 3),
        np.float32,
        minimum=[-0.1, 2.0, -100.0],
        maximum=[0.2, 2.0, 0.0],
    )
    sample = spec.sample(np.random.default_rng(14))
    spec.validate(sample)
    self.assertTrue(np.all(sample[:, 1] == 2.0))
    self.assertTrue(np.all(sample[:, 0] >= -0.1))
    self.assertTrue(np.all(sample[:, 0] <= 0.2))
    self.assertGreater(np.unique(sample[:, 0]).size, 100)

  def test_floating_infinite_limits_produce_finite_values(self):
    spec = specs.BoundedArray(
        (300, 4),
        np.float64,
        minimum=[-np.inf, 2.0, -np.inf, -3.0],
        maximum=[np.inf, np.inf, -2.0, 3.0],
    )
    sample = spec.sample(np.random.default_rng(9))
    spec.validate(sample)
    self.assertTrue(np.isfinite(sample).all())
    self.assertTrue(np.all(sample[:, 1] >= 2))
    self.assertTrue(np.all(sample[:, 2] <= -2))
    self.assertTrue(np.all(sample[:, 3] <= 3))

  def test_fixed_infinite_float_bounds_are_exact(self):
    for value in (np.inf, -np.inf):
      with self.subTest(value=value):
        spec = specs.BoundedArray((), np.float64, value, value)
        sample = spec.sample(np.random.default_rng(2))
        spec.validate(sample)
        self.assertEqual(sample.item(), value)

  def test_boolean_boundaries_and_discrete_actions(self):
    for spec in (
        specs.Array((32,), np.bool_),
        specs.BoundedArray((32,), np.bool_, minimum=False, maximum=True),
        specs.DiscreteArray(4, dtype=np.int32),
    ):
      with self.subTest(spec=spec):
        sample = spec.sample(np.random.default_rng(24))
        spec.validate(sample)
    fixed = specs.BoundedArray((5,), np.bool_, minimum=True, maximum=True)
    self.assertTrue(fixed.sample().all())

  def test_discrete_array_uses_inclusive_action_range(self):
    spec = specs.DiscreteArray(5, dtype=np.int8)
    values = [
        int(spec.sample(np.random.default_rng(seed))) for seed in range(80)
    ]
    self.assertEqual(min(values), 0)
    self.assertEqual(max(values), 4)
    self.assertTrue(all(0 <= value < 5 for value in values))

  @parameterized.named_parameters(('str', str), ('bytes', bytes))
  def test_string_array_preserves_element_type(self, string_type):
    spec = specs.StringArray((3, 4), string_type=string_type)
    sample = spec.sample(np.random.default_rng(21))
    spec.validate(sample)
    self.assertTrue(all(isinstance(item, string_type) for item in sample.flat))
    self.assertTrue(all(bool(item) for item in sample.flat))
    self.assertGreater(len(set(sample.flat)), 1)

  def test_string_array_scalar_and_empty_shapes(self):
    for shape in ((), (0,), (2, 0)):
      with self.subTest(shape=shape):
        spec = specs.StringArray(shape, bytes)
        result = spec.sample(np.random.default_rng(6))
        spec.validate(result)
        self.assertEqual(result.shape, shape)

  def test_sampling_does_not_change_existing_generate_value(self):
    spec = specs.BoundedArray((2,), np.int32, 3, 7)
    np.testing.assert_array_equal(spec.generate_value(), [3, 3])
    _ = spec.sample(np.random.default_rng(5))
    np.testing.assert_array_equal(spec.generate_value(), [3, 3])


if __name__ == '__main__':
  absltest.main()

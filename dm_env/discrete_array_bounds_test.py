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
"""Checks that DiscreteArray validates integer values rather than dtype order."""

import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np

_INTEGER_DTYPES = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
                   np.int64, np.uint64)


class DiscreteArrayBoundsTest(parameterized.TestCase):

  @parameterized.product(dtype=_INTEGER_DTYPES, excess=(1, 17))
  def test_rejects_unrepresentable_maximum(self, dtype, excess):
    num_values = int(np.iinfo(dtype).max) + 1 + excess
    message = specs._DTYPE_OVERFLOW.format(np.dtype(dtype), num_values)
    with self.assertRaisesWithLiteralMatch(ValueError, message):
      specs.DiscreteArray(num_values, dtype=dtype)

  @parameterized.product(dtype=_INTEGER_DTYPES, full_range=(False, True))
  def test_representable_endpoints_and_round_trip(self, dtype, full_range):
    maximum = int(np.iinfo(dtype).max) if full_range else 0
    spec = specs.DiscreteArray(maximum + 1, dtype=dtype, name='action')
    self.assertEqual(spec.num_values, maximum + 1)
    self.assertEqual(int(spec.maximum), maximum)
    for value in (0, maximum):
      actual = spec.validate(np.asarray(value, dtype=dtype))
      self.assertEqual(int(actual), value)
      self.assertEqual(actual.dtype, np.dtype(dtype))
    spec.validate(spec.generate_value())
    restored = pickle.loads(pickle.dumps(spec))
    self.assertEqual(restored, spec)
    self.assertEqual(restored.num_values, spec.num_values)
    self.assertEqual(restored.name, 'action')

  @parameterized.parameters('int8', np.dtype('int8'), np.dtype('>i2'))
  def test_dtype_objects_and_strings_share_overflow_validation(self, dtype):
    num_values = int(np.iinfo(dtype).max) + 2
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._DTYPE_OVERFLOW.format(np.dtype(dtype), num_values)):
      specs.DiscreteArray(num_values, dtype=dtype)

  @parameterized.parameters(np.int32(129), np.uint64(129))
  def test_numpy_integer_counts_use_the_same_check(self, num_values):
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._DTYPE_OVERFLOW.format(np.dtype('int8'), num_values)):
      specs.DiscreteArray(num_values, dtype=np.int8)

  def test_replacing_dtype_rejects_a_range_that_no_longer_fits(self):
    original = specs.DiscreteArray(256, np.uint8)
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._DTYPE_OVERFLOW.format(np.dtype('int8'), 256)):
      original.replace(dtype=np.int8)
    self.assertEqual(original.dtype, np.dtype('uint8'))
    self.assertEqual(int(original.maximum), 255)
    self.assertEqual(int(original.replace(dtype=np.int16).maximum), 255)


if __name__ == '__main__':
  absltest.main()

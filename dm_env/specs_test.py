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
"""Tests for dm_env.specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np
import six
from six.moves import cPickle as pickle


class ArrayTest(parameterized.TestCase):

  def testShapeTypeError(self):
    with self.assertRaises(TypeError):
      specs.Array(32, np.int32)

  def testShapeElementTypeError(self):
    with self.assertRaises(TypeError):
      specs.Array([None], np.int32)

  def testDtypeTypeError(self):
    with self.assertRaises(TypeError):
      specs.Array((1, 2, 3), "32")

  def testScalarShape(self):
    specs.Array((), np.int32)

  def testStringDtype(self):
    specs.Array((1, 2, 3), "int32")

  def testNumpyDtype(self):
    specs.Array((1, 2, 3), np.int32)

  def testDtype(self):
    spec = specs.Array((1, 2, 3), np.int32)
    self.assertEqual(np.int32, spec.dtype)

  def testShape(self):
    spec = specs.Array([1, 2, 3], np.int32)
    self.assertEqual((1, 2, 3), spec.shape)

  def testEqual(self):
    spec_1 = specs.Array((1, 2, 3), np.int32)
    spec_2 = specs.Array((1, 2, 3), np.int32)
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentShape(self):
    spec_1 = specs.Array((1, 2, 3), np.int32)
    spec_2 = specs.Array((1, 3, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualDifferentDtype(self):
    spec_1 = specs.Array((1, 2, 3), np.int64)
    spec_2 = specs.Array((1, 2, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = specs.Array((1, 2, 3), np.int32)
    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testIsUnhashable(self):
    spec = specs.Array(shape=(1, 2, 3), dtype=np.int32)
    with six.assertRaisesRegex(self, TypeError, "unhashable type"):
      hash(spec)

  @parameterized.parameters(
      dict(value=np.zeros((1, 2), dtype=np.int32), is_valid=True),
      dict(value=np.zeros((1, 2), dtype=np.float32), is_valid=False),
  )
  def testValidateDtype(self, value, is_valid):
    spec = specs.Array((1, 2), np.int32)
    if is_valid:  # Should not raise any exception.
      spec.validate(value)
    else:
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          specs._INVALID_DTYPE % (spec.dtype, value.dtype)):
        spec.validate(value)

  @parameterized.parameters(
      dict(value=np.zeros((1, 2), dtype=np.int32), is_valid=True),
      dict(value=np.zeros((1, 2, 3), dtype=np.int32), is_valid=False),
  )
  def testValidateShape(self, value, is_valid):
    spec = specs.Array((1, 2), np.int32)
    if is_valid:  # Should not raise any exception.
      spec.validate(value)
    else:
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          specs._INVALID_SHAPE % (spec.shape, value.shape)):
        spec.validate(value)

  def testGenerateValue(self):
    spec = specs.Array((1, 2), np.int32)
    test_value = spec.generate_value()
    spec.validate(test_value)

  def testSerialization(self):
    desc = specs.Array([1, 5], np.float32, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @parameterized.parameters(
      {"arg_name": "shape", "new_value": (2, 3)},
      {"arg_name": "dtype", "new_value": np.int32},
      {"arg_name": "name", "new_value": "something_else"})
  def testReplace(self, arg_name, new_value):
    old_spec = specs.Array([1, 5], np.float32, "test")
    new_spec = old_spec.replace(**{arg_name: new_value})
    self.assertIsNot(old_spec, new_spec)
    self.assertEqual(getattr(new_spec, arg_name), new_value)
    for attr_name in set(["shape", "dtype", "name"]).difference([arg_name]):
      self.assertEqual(getattr(new_spec, attr_name),
                       getattr(old_spec, attr_name))

  def testReplaceRaisesTypeErrorIfSubclassAcceptsVarArgs(self):

    class InvalidSpecSubclass(specs.Array):

      def __init__(self, *args):  # pylint: disable=useless-super-delegation
        super(InvalidSpecSubclass, self).__init__(*args)

    spec = InvalidSpecSubclass([1, 5], np.float32, "test")

    with self.assertRaisesWithLiteralMatch(
        TypeError, specs._VAR_ARGS_NOT_ALLOWED):
      spec.replace(name="something_else")

  def testReplaceRaisesTypeErrorIfSubclassAcceptsVarKwargs(self):

    class InvalidSpecSubclass(specs.Array):

      def __init__(self, **kwargs):  # pylint: disable=useless-super-delegation
        super(InvalidSpecSubclass, self).__init__(**kwargs)

    spec = InvalidSpecSubclass(shape=[1, 5], dtype=np.float32, name="test")

    with self.assertRaisesWithLiteralMatch(
        TypeError, specs._VAR_KWARGS_NOT_ALLOWED):
      spec.replace(name="something_else")


class BoundedArrayTest(parameterized.TestCase):

  def testInvalidMinimum(self):
    with six.assertRaisesRegex(self, ValueError, "not compatible"):
      specs.BoundedArray((3, 5), np.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    with six.assertRaisesRegex(self, ValueError, "not compatible"):
      specs.BoundedArray((3, 5), np.uint8, 0, (1, 1, 1))

  def testMinMaxAttributes(self):
    spec = specs.BoundedArray((1, 2, 3), np.float32, 0, (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)

  @parameterized.parameters(
      dict(spec_dtype=np.float32, min_dtype=np.float64, max_dtype=np.int32),
      dict(spec_dtype=np.uint64, min_dtype=np.uint8, max_dtype=float))
  def testMinMaxCasting(self, spec_dtype, min_dtype, max_dtype):
    minimum = np.array(0., dtype=min_dtype)
    maximum = np.array((3.14, 15.9, 265.4), dtype=max_dtype)
    spec = specs.BoundedArray(
        shape=(1, 2, 3), dtype=spec_dtype, minimum=minimum, maximum=maximum)
    self.assertEqual(spec.minimum.dtype, spec_dtype)
    self.assertEqual(spec.maximum.dtype, spec_dtype)

  def testReadOnly(self):
    spec = specs.BoundedArray((1, 2, 3), np.float32, 0, (5, 5, 5))
    with six.assertRaisesRegex(self, ValueError, "read-only"):
      spec.minimum[0] = -1
    with six.assertRaisesRegex(self, ValueError, "read-only"):
      spec.maximum[0] = 100

  def testEqualBroadcastingBounds(self):
    spec_1 = specs.BoundedArray(
        (1, 2), np.float32, minimum=0.0, maximum=1.0)
    spec_2 = specs.BoundedArray(
        (1, 2), np.float32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentMinimum(self):
    spec_1 = specs.BoundedArray(
        (1, 2), np.float32, minimum=[0.0, -0.6], maximum=[1.0, 1.0])
    spec_2 = specs.BoundedArray(
        (1, 2), np.float32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = specs.BoundedArray(
        (1, 2), np.float32, minimum=[0.0, -0.6], maximum=[1.0, 1.0])
    spec_2 = specs.Array((1, 2), np.float32)
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testNotEqualDifferentMaximum(self):
    spec_1 = specs.BoundedArray(
        (1, 2), np.int32, minimum=0.0, maximum=2.0)
    spec_2 = specs.BoundedArray(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testIsUnhashable(self):
    spec = specs.BoundedArray(
        shape=(1, 2), dtype=np.int32, minimum=0.0, maximum=2.0)
    with six.assertRaisesRegex(self, TypeError, "unhashable type"):
      hash(spec)

  def testRepr(self):
    as_string = repr(specs.BoundedArray(
        (1, 2), np.int32, minimum=101.0, maximum=73.0))
    self.assertIn("101", as_string)
    self.assertIn("73", as_string)

  @parameterized.parameters(
      dict(value=np.array([[5, 6], [8, 10]], dtype=np.int32), is_valid=True),
      dict(value=np.array([[5, 6], [8, 11]], dtype=np.int32), is_valid=False),
      dict(value=np.array([[4, 6], [8, 10]], dtype=np.int32), is_valid=False),
  )
  def testValidateBounds(self, value, is_valid):
    spec = specs.BoundedArray((2, 2), np.int32, minimum=5, maximum=10)
    if is_valid:  # Should not raise any exception.
      spec.validate(value)
    else:
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          specs._OUT_OF_BOUNDS % (spec.minimum, value, spec.maximum)):
        spec.validate(value)

  def testValidateReturnsValue(self):
    spec = specs.BoundedArray([1], np.int32, minimum=0, maximum=1)
    validated_value = spec.validate(np.array([0], dtype=np.int32))
    self.assertIsNotNone(validated_value)

  def testGenerateValue(self):
    spec = specs.BoundedArray((2, 2), np.int32, minimum=5, maximum=10)
    test_value = spec.generate_value()
    spec.validate(test_value)

  def testScalarBounds(self):
    spec = specs.BoundedArray((), np.float, minimum=0.0, maximum=1.0)

    self.assertIsInstance(spec.minimum, np.ndarray)
    self.assertIsInstance(spec.maximum, np.ndarray)

    # Sanity check that numpy compares correctly to a scalar for an empty shape.
    self.assertEqual(0.0, spec.minimum)
    self.assertEqual(1.0, spec.maximum)

    # Check that the spec doesn't fail its own input validation.
    _ = specs.BoundedArray(
        spec.shape, spec.dtype, spec.minimum, spec.maximum)

  def testSerialization(self):
    desc = specs.BoundedArray([1, 5], np.float32, -1, 1, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @parameterized.parameters(
      {"arg_name": "shape", "new_value": (2, 3)},
      {"arg_name": "dtype", "new_value": np.int32},
      {"arg_name": "name", "new_value": "something_else"},
      {"arg_name": "minimum", "new_value": -2},
      {"arg_name": "maximum", "new_value": 2},
  )
  def testReplace(self, arg_name, new_value):
    old_spec = specs.BoundedArray([1, 5], np.float32, -1, 1, "test")
    new_spec = old_spec.replace(**{arg_name: new_value})
    self.assertIsNot(old_spec, new_spec)
    self.assertEqual(getattr(new_spec, arg_name), new_value)
    for attr_name in set(["shape", "dtype", "name", "minimum", "maximum"]
                        ).difference([arg_name]):
      self.assertEqual(getattr(new_spec, attr_name),
                       getattr(old_spec, attr_name))


class DiscreteArrayTest(parameterized.TestCase):

  @parameterized.parameters(0, -3)
  def testInvalidNumActions(self, num_values):
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._NUM_VALUES_NOT_POSITIVE.format(num_values)):
      specs.DiscreteArray(num_values=num_values)

  @parameterized.parameters(np.float32, np.object)
  def testDtypeNotIntegral(self, dtype):
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._DTYPE_NOT_INTEGRAL.format(dtype)):
      specs.DiscreteArray(num_values=5, dtype=dtype)

  @parameterized.parameters(
      dict(dtype=np.uint8, num_values=2 ** 8 + 1),
      dict(dtype=np.uint64, num_values=2 ** 64 + 1))
  def testDtypeOverflow(self, num_values, dtype):
    with self.assertRaisesWithLiteralMatch(
        ValueError, specs._DTYPE_OVERFLOW.format(np.dtype(dtype), num_values)):
      specs.DiscreteArray(num_values=num_values, dtype=dtype)

  def testRepr(self):
    as_string = repr(specs.DiscreteArray(num_values=5))
    self.assertIn("num_values=5", as_string)

  def testProperties(self):
    num_values = 5
    spec = specs.DiscreteArray(num_values=5)
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, num_values - 1)
    self.assertEqual(spec.dtype, np.int32)
    self.assertEqual(spec.num_values, num_values)

  def testSerialization(self):
    desc = specs.DiscreteArray(2, np.int32, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @parameterized.parameters(
      {"arg_name": "num_values", "new_value": 4},
      {"arg_name": "dtype", "new_value": np.int64},
      {"arg_name": "name", "new_value": "something_else"})
  def testReplace(self, arg_name, new_value):
    old_spec = specs.DiscreteArray(2, np.int32, "test")
    new_spec = old_spec.replace(**{arg_name: new_value})
    self.assertIsNot(old_spec, new_spec)
    self.assertEqual(getattr(new_spec, arg_name), new_value)
    for attr_name in set(
        ["num_values", "dtype", "name"]).difference([arg_name]):
      self.assertEqual(getattr(new_spec, attr_name),
                       getattr(old_spec, attr_name))

if __name__ == "__main__":
  absltest.main()

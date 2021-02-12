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
"""Base class for TestMixin classes."""


class TestMixin:
  """Base class for TestMixins.

  Subclasses must override `make_object_under_test`.
  """

  def setUp(self):
    # A call to super is required for cooperative multiple inheritance to work.
    super().setUp()
    test_method = getattr(self, self._testMethodName)
    make_obj_kwargs = getattr(test_method, "_make_obj_kwargs", {})
    self.object_under_test = self.make_object_under_test(**make_obj_kwargs)

  def make_object_under_test(self, **unused_kwargs):
    raise NotImplementedError(
        "Attempt to run tests from an abstract TestMixin subclass %s. "
        "Perhaps you forgot to override make_object_under_test?" % type(self))

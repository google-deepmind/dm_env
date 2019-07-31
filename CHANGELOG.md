# Changelog

All significant changes to this project will be documented here.

## [Unreleased]

### Added

*   Specs now have a `replace` method that can be used to create a new instance
    with some of the attributes replaced (similar to `namedtuple._replace`).

### Changed

*   The `BoundedArray` constructor now casts `minimum` and `maximum` so that
    their dtypes match that of the spec instance.

## [1.0]

Release date: 2019-07-18

*   Initial release.

[Unreleased]: https://github.com/deepmind/dm_env/compare/v1.0...HEAD
[1.0]: https://github.com/deepmind/dm_env/releases/tag/v1.0

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added

- Initial release
- `IRNode` enum with operators: `Scan`, `Map`, `Filter`, `Join`, `Distinct`, `Union`
- `Predicate` enum for filter conditions with comparison and logical operators
- Schema tracking via `output_schema()` method
- Pretty printing for debugging with `pretty_print()`
- Predicate utilities:
  - `referenced_columns()` - find all columns used in a predicate
  - `simplify()` - constant folding optimization
  - `adjust_for_projection()` - reindex columns after projection

[Unreleased]: https://github.com/jsam/datalog_ir/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jsam/datalog_ir/releases/tag/v0.1.0

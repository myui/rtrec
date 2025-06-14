# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025.06.14

### Bug Fixes

* Fix `@override` decorator import compatibility for Python 3.10-3.11 by using conditional imports with `typing_extensions` fallback

**Full Changelog**: [v0.2.2...v0.2.3](https://github.com/myui/rtrec/compare/v0.2.2...v0.2.3)

## [0.2.2] - 2025.06.14

This release improves Python version compatibility by adding proper support for the `Self` type hint across Python 3.10+.

### Improvements

* Add support for 'Self' type in type hints for compatibility with Python 3.10+ ([c22b4b0](https://github.com/myui/rtrec/commit/c22b4b0))

**Full Changelog**: [v0.2.1...v0.2.2](https://github.com/myui/rtrec/compare/v0.2.1...v0.2.2)

## [0.2.1] - 2025.06.13

This release focuses on improving the library's modularity by making web serving components optional, enhancing type safety, and adding essential model persistence capabilities.

### Improvements

* Refactor dependencies: move FastAPI, Pydantic, Uvicorn, HTTPX to optional dependencies and add Boto3 to dev-dependencies ([586bb56](https://github.com/myui/rtrec/commit/586bb56))

### Enhancements

* Implement model serialization and deserialization methods for BaseModel and its subclasses ([#5](https://github.com/myui/rtrec/pull/5))
* Type safety improvements ([#6](https://github.com/myui/rtrec/pull/6))

### Bug Fixes

* Fix type errors ([#3](https://github.com/myui/rtrec/pull/3), [#4](https://github.com/myui/rtrec/pull/4))

**Full Changelog**: [v0.2.0...v0.2.1](https://github.com/myui/rtrec/compare/v0.2.0...v0.2.1)

## [0.2.0] - 05.04.2025

This release introduces architectural improvements to the recommendation system core, with a focus on separating recommendation logic for hot and cold users.

### Major Changes

* Separated recommendation logic for hot and cold users ([04bcae3](https://github.com/myui/rtrec/commit/04bcae38f52c6d95541d4097edaf37c8e16745c3))
* Added a notebook demonstrating Hybrid model for H&M fashion recommendations ([64a5da2](https://github.com/myui/rtrec/commit/64a5da2850b5e21fbd2d2947c0996ed1f967d33a))

### Minor Changes

* Added sort_by_tstamp option for time-based recommendation ordering ([18e1e34](https://github.com/myui/rtrec/commit/18e1e342f983915f952513dfe851c370c1c9f6d0))

### Bug Fixes

* Fixed corner case for handling not yet learned features ([2801a65](https://github.com/myui/rtrec/commit/2801a65e240082caa295c94579b3c08fe9ed716c))

### Breaking Changes

* Redesigned recommendation batch processing with separate paths for cold and hot users, which may require updates to custom model implementations

**Full Changelog**: [v0.1.9...v0.2.0](https://github.com/myui/rtrec/compare/v0.1.9...v0.2.0)

## [0.1.9] - 04.26.2025

This release adds support for additional datasets, introduces a new hybrid model, and includes several improvements and bug fixes.

### Major changes

* Added new HybridSlimFM model combining FM (content-based) with SLIM (collaborative filtering) ([Commit 1189a57](https://github.com/myui/rtrec/commit/1189a57175bb74b21823faf047932489a09d8bfe))
* Added support for H&M Kaggle dataset ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))
* Added support for RetailRocket dataset ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))

### Improvements

* Added handle_unknown_user hook for handling cold start users in LightFM ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))
* Added force_identify option to control identity handling ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))
* Added user_column and tstamp_column arguments for flexible schema handling ([Commit ebb2579](https://github.com/myui/rtrec/commit/ebb2579d0b1dd41f874076cf959002626dfafb85))
* Supported ret_scores argument in recommend() method ([Commit 24a44e6](https://github.com/myui/rtrec/commit/24a44e656a860f9206f340cef74a01fabfc4d280))
* Improved memory usage in interaction_counts ([Commit fb02997](https://github.com/myui/rtrec/commit/fb02997ed46dd3e2a4deef0f412bac6790cc9c27))

### Bug fixes

* Fixed a bug when slicing scores ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))
* Fixed URL for Amazon dataset ([Commit 673f02e](https://github.com/myui/rtrec/commit/673f02e17b45ba7b82a31c42a24ba9e05de821e9))
* Fixed type hint issues ([Commit 2a7a69f](https://github.com/myui/rtrec/commit/2a7a69f12a59c6ccae3c7d6e0d76e2d3faacf076))

### Other changes

* Re-enabled @override decorator ([Commit ed97524](https://github.com/myui/rtrec/commit/ed9752413b21fba8186cb7fd2b3284f0284a6ec9))
* Removed deprecated functionality ([Commit 6f1daa7](https://github.com/myui/rtrec/commit/6f1daa785e997f872c467f254bc5c148548f10b3))

**Full Changelog**: [v0.1.8...v0.1.9](https://github.com/myui/rtrec/compare/v0.1.8...v0.1.9)

## [0.1.8] - 03.24.2025

Minor enhancement release.

### Minor changes

* Revised to return recent hot items for U2I recommendation where user does not have any interaction ([Commit 1f383b7](https://github.com/myui/rtrec/commit/1f383b755093fc9636480bc8166fb68ccff615e8
))

* Reduced dependency `@override` in typing-extension >= 4.5.0 ([Commit 48125c2](https://github.com/myui/rtrec/commit/48125c25bed0340852cf749587607842cb62b34d))

* Suppported SGD-based optimization for SLIM ([Commit ecf5d29](https://github.com/myui/rtrec/commit/ecf5d2942a8649005b1ffbad87391356e52182f0))

### Bug fixes

* Fixed a corner case bug for handling candidate_item_ids ([Commit f6231ed](https://github.com/myui/rtrec/commit/f6231ed51816a2e1e11954bbad0becfc56992e61))

**Full Changelog**: [v0.1.7...v0.1.8](https://github.com/myui/rtrec/compare/v0.1.7...v0.1.8)

## [0.1.7] - 01.17.2025

All users of v0.1.6 is recommended to update to v0.1.7.
LightFM support is now bacame stable.

**Full Changelog**: [v0.1.6...v0.1.7](https://github.com/myui/rtrec/compare/v0.1.6...v0.1.7)

### Major changes

* Supported Context Features (user/item tags) for LightFM model.

Check [this example notebook](https://github.com/myui/rtrec/blob/main/notebooks/rtrec-movielens-with-features.ipynb) how to use user/item tags.

## [0.1.6] - 01.06.2025

All users of 0.1.5 is recommended to update to v0.1.6.

**Full Changelog**: [v0.1.5...v0.1.6](https://github.com/myui/rtrec/compare/v0.1.5...v0.1.6)

### Bug fixes

* Fixed a bug in LightFM models.

* Fixed a bug in similar_items() method ([Commit d8e2a98](https://github.com/myui/rtrec/commit/d8e2a98e6ef757259b8bd874c1c7464de1830dc6))

### Minor Changes

* Supported returning scores with ret_scores argument for similar_items() ([Commit b824a52](https://github.com/myui/rtrec/commit/b824a525708d6c704f7213b8c10b7fb5801dead3))

## [0.1.5] - 12.27.2004

### Bug fixes 

* Fixed a bug for max\_user\_id/max\_item\_id ([Commit 72543b6](https://github.com/myui/rtrec/commit/72543b6f7956f49a960a7ded382c3d7f5752e8d7))

## [0.1.4] - 12.27.2024

### Major changes

* Supported [LightFM](https://github.com/lyst/lightfm)

### Minor Changes

* Added [recommend\_batch](https://github.com/myui/rtrec/commit/7f7f857e810e374648081770eca00743579247c9)

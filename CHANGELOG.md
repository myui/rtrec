# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

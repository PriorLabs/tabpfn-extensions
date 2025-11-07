# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

## [0.2.1] - 2025-11-07

### Added

### Changed
- The search space for tuning and AutoTabPFN has been updated to work again with tabpfn 6.0 in #196.

## [0.2.0] - 2025-11-06

### Added

### Changed
- Updated tabpfn package to 6.0.5. This means **for most extensions, the new v2.5 model will be
  used** (excluding AutoTabPFN and tuned for now).
- Fixed type mistmatch in imputation #195
- Fixed SurvivalTabPFN documentation and clean ranking logic #194

## [0.1.6] - 2025-10-02

### Added
- Allow users to opt in for extended anonymous analytics and our newsletter

### Changed

## [0.1.5] - 2025-09-19

### Added
- `TabEBM` class for synthetic tabular data generation using SGLD
- `partial_dependence_plots` for visualizing feature influence (supports PDP and ICE via scikit-learn’s `PartialDependenceDisplay`)
- Usage analytics (anonymous, optional, enabled by default)

### Changed

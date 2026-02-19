# Benchmark Project Cursor Skills

This directory contains 6 specialized Cursor skills designed for the image/video augmentation benchmark project.

## Available Skills

### 1. [Benchmark Runner](benchmark-runner/SKILL.md)
**Triggers**: benchmark, run_all.sh, run_single.sh, performance testing

Automates running benchmarks with standard configurations:
- Run single or multiple library benchmarks
- Validate JSON outputs
- Generate comparison reports and plots
- Update documentation automatically

### 2. [Library Integration](library-integration/SKILL.md)
**Triggers**: add library, integrate library, new augmentation library

Guides adding new augmentation libraries:
- Create transform implementation files
- Set up requirements and virtual environments
- Update run scripts
- Generate baseline results

### 3. [Pre-commit Validation](pre-commit-validation/SKILL.md)
**Triggers**: commit, pre-commit, ruff, mypy, pyright, linter

Ensures code quality before committing:
- Run and fix ruff/mypy/pyright issues
- Auto-format code
- Validate type annotations
- Iterative error fixing

### 4. [Performance Analysis](performance-analysis/SKILL.md)
**Triggers**: performance, analyze results, slow benchmark, regression

Deep analysis of benchmark results:
- Identify slow transforms and warmup issues
- Compare speedups across libraries
- Detect performance regressions
- Generate optimization recommendations

### 5. [Documentation Generator](documentation-generator/SKILL.md)
**Triggers**: update docs, documentation, update_docs.sh, generate tables

Automates documentation updates:
- Generate comparison tables
- Create speedup plots
- Update README summaries
- Maintain library metadata

### 6. [Transform Spec Validator](transform-spec-validator/SKILL.md)
**Triggers**: validate transforms, custom transforms, spec file

Validates custom transform specifications:
- Check LIBRARY, __call__, TRANSFORMS structure
- Test transforms on sample data
- Verify naming conventions
- Debug common issues

## Quick Start

These skills are automatically available to Cursor's AI agent. The agent will use them when relevant based on your requests and the trigger terms.

### Example Usage

```bash
# The agent will use benchmark-runner skill
"Run benchmarks for all libraries"

# The agent will use pre-commit-validation skill
"Fix linter errors and commit"

# The agent will use performance-analysis skill
"Why is this transform so slow?"

# The agent will use library-integration skill
"Add support for torchvision transforms"
```

## Skill Maintenance

To update a skill:
1. Edit the `SKILL.md` file in the skill's directory
2. Keep descriptions specific and include trigger terms
3. Maintain concise instructions (< 500 lines preferred)
4. Test with actual use cases

## Project Context

These skills are tailored for a benchmark suite that:
- Tests 5+ augmentation libraries (AlbumentationsX, imgaug, torchvision, Kornia, Augly)
- Uses Python 3.13+ with strict type checking
- Manages separate virtual environments per library
- Generates performance comparisons and documentation
- Uses pre-commit hooks for code quality

For more details, see the main [README.md](../../README.md).

---
name: pre-commit-validation
description: Runs pre-commit hooks on changed files, fixes common ruff/mypy/pyright issues, and validates code quality. Use when committing code, before pushing, when fixing linter errors, or when the user mentions pre-commit, ruff, mypy, or pyright.
---

# Pre-commit Validation

Ensure code passes all quality checks before committing.

## Quick Run

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files benchmark/runner.py tools/compare_results.py

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

## Auto-fix Common Issues

Most issues can be fixed automatically:

```bash
# Ruff auto-fixes
pre-commit run ruff --all-files
# Runs: ruff check --fix

# Format code
pre-commit run ruff-format --all-files

# Fix trailing whitespace, EOF, etc.
pre-commit run trailing-whitespace --all-files
pre-commit run end-of-file-fixer --all-files
```

## Type Checking

### Mypy
Checks `benchmark/` and `tools/` directories:

```bash
pre-commit run mypy --all-files
```

Common mypy fixes:
- Add type annotations to function signatures
- Use `from typing import Any` for flexible types
- Use `# type: ignore[error-code]` for unavoidable issues

### Pyright
Stricter type checking:

```bash
pre-commit run pyright --all-files
```

Config in `pyproject.toml`:
- `pythonVersion = "3.13"`
- `reportMissingImports = false` (for optional deps)

## Project Quality Standards

### Ruff Configuration
- Line length: 120
- Target: Python 3.13
- Format: double quotes, 4-space indent
- Ignores: D100-D107 (docstrings), ANN001/201 (some type annotations)

### File Scope
Type checkers only run on:
- `benchmark/**/*.py`
- `tools/**/*.py`

Not checked:
- `examples/`
- Test files
- Scripts in root

## Validation Loop

When fixing issues iteratively:

1. **Run pre-commit**: `pre-commit run --all-files`
2. **Review errors**: Note specific files/lines
3. **Fix issues**: Auto-fix or manual correction
4. **Re-run**: Verify fixes worked
5. **Repeat**: Until all checks pass

## Common Errors and Fixes

### Ruff Errors

**Unused imports**:
```python
# Error: F401 'numpy' imported but unused
import numpy as np  # Remove or use

# Fix: Remove if truly unused
```

**Line too long**:
```python
# Error: E501 line too long (125 > 120)

# Fix: Break into multiple lines
result = some_function(
    arg1, arg2, arg3
)
```

### Mypy Errors

**Missing return type**:
```python
# Error: Function is missing a return type annotation

# Fix: Add return type
def process_data(x: int) -> dict[str, Any]:
    return {"result": x}
```

**Type mismatch**:
```python
# Error: Argument has incompatible type

# Fix: Cast or change type
value: str = str(numeric_value)
```

### Pyright Errors

Similar to mypy but stricter. Often requires:
- More explicit type annotations
- Better handling of Optional types
- Explicit type guards

## Skip Checks (Rarely Needed)

Only skip when absolutely necessary:

```python
# Skip specific error
x = problematic_code()  # type: ignore[error-code]

# Skip all on line (avoid)
x = problematic_code()  # type: ignore
```

For ruff:
```python
# noqa: E501
long_line_that_cannot_be_broken = "..."
```

## Integration with Workflow

Before committing:
```bash
# 1. Check what will be committed
git status
git diff

# 2. Run pre-commit
pre-commit run --all-files

# 3. If errors, fix and re-run
# ... fix issues ...
pre-commit run --all-files

# 4. Commit when clean
git add .
git commit -m "feat: add new feature"
```

## CI Integration

Pre-commit runs automatically in CI (`.github/workflows/ci.yml`):
- On pull requests to main
- Python 3.13
- All files checked

Ensure local pre-commit passes before pushing to avoid CI failures.

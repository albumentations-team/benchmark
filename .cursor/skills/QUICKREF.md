# Cursor Skills Quick Reference

## Testing the Skills

### 1. Test Benchmark Runner
```bash
# Ask Cursor: "Show me how to run benchmarks for albumentationsx"
# Expected: Should reference run_single.sh with proper parameters
```

### 2. Test Pre-commit Validation
```bash
# Ask Cursor: "Check if code passes pre-commit"
# Expected: Should run pre-commit run --all-files
```

### 3. Test Performance Analysis
```bash
# Ask Cursor: "Why is MedianBlur so slow?"
# Expected: Should load JSON and check time_per_image metrics
```

### 4. Test Library Integration
```bash
# Ask Cursor: "How do I add support for a new library called 'newlib'?"
# Expected: Should outline creating _impl.py, requirements.txt, updating run scripts
```

### 5. Test Documentation Generator
```bash
# Ask Cursor: "Update benchmark documentation"
# Expected: Should reference update_docs.sh
```

### 6. Test Transform Spec Validator
```bash
# Ask Cursor: "Validate my custom transforms file"
# Expected: Should check LIBRARY, __call__, TRANSFORMS structure
```

## Skill File Structure

Each skill follows this pattern:

```markdown
---
name: skill-name
description: What it does and when to use it (includes trigger terms)
---

# Skill Title

## Quick Start
[Most common usage]

## Detailed Instructions
[Step-by-step guidance]

## Common Issues
[Troubleshooting]
```

## Updating Skills

To modify a skill:

1. Edit `.cursor/skills/{skill-name}/SKILL.md`
2. Keep under 500 lines if possible
3. Update description if triggers change
4. Add examples for clarity

## Skill Statistics

| Skill | Lines | Key Focus |
|-------|-------|-----------|
| benchmark-runner | 114 | Running benchmarks |
| documentation-generator | 228 | Updating docs |
| library-integration | 185 | Adding libraries |
| performance-analysis | 224 | Analyzing results |
| pre-commit-validation | 187 | Code quality |
| transform-spec-validator | 303 | Validating specs |

All skills are well under the recommended 500 line limit.

## Benefits

These skills enable the AI agent to:
- **Reduce token usage**: Pre-written instructions vs generating from scratch
- **Ensure consistency**: Same approach every time
- **Capture domain knowledge**: Project-specific patterns and conventions
- **Speed up development**: Agent knows exactly what to do
- **Prevent errors**: Built-in validation and best practices

## Next Steps

1. **Test the skills**: Try the test commands above
2. **Refine as needed**: Update based on actual usage
3. **Add examples**: Put sample outputs in skill directories if helpful
4. **Monitor effectiveness**: Track how often skills are triggered

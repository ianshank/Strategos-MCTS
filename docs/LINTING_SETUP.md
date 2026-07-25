# Linting and Formatting Setup

## 🎯 Goal: Zero-Friction Development

This project now has **automated linting and formatting** to prevent CI failures from formatting issues.

---

## 🛠️ Setup (One-Time)

### Install Development Tools

```bash
pip install ruff black pre-commit
```

Or install all dev dependencies:

```bash
pip install -e ".[dev]"
```

---

## 🚀 Usage

### Option 1: Automatic (Recommended)

The **pre-push hook** automatically formats your code before pushing:

```bash
git push  # Auto-formats and fixes linting issues
```

### Option 2: Manual

Run the comprehensive linting script:

```bash
# Auto-fix everything
python scripts/lint_and_format.py

# Check only (no fixes)
python scripts/lint_and_format.py --check
```

### Option 3: Direct Commands

```bash
# Format code
black . --line-length 120

# Fix linting issues
ruff check . --fix

# Check without fixing
ruff check .
```

---

## 📋 What Gets Checked

### 1. **Code Formatting** (Ruff Format)
- Line length: 120 characters
- Double quotes for strings
- 4-space indentation
- Consistent spacing

### 2. **Linting** (Ruff Check)
- **E/W**: PEP 8 style errors and warnings
- **F**: Pyflakes (unused imports, undefined names)
- **I**: Import sorting (isort)
- **B**: Bugbear (common bugs and anti-patterns)
- **C4**: Comprehension improvements
- **UP**: Modern Python syntax (pyupgrade)

### 3. **Ignored Rules** (Overly Strict)
The following rules are **disabled** to reduce friction:
- `ARG002`: Unused method arguments (often required by protocols)
- `SIM117`: Combine with statements (sometimes clearer separate)
- `F541`: F-strings without placeholders
- `B905`: zip() without strict parameter
- And more... (see `pyproject.toml`)

---

## 🔄 CI Workflow

### Automatic Formatting in CI

The CI pipeline now **automatically fixes** formatting and linting issues:

1. **Auto-format** code with `black`
2. **Auto-fix** linting issues with `ruff check --fix`
3. **Auto-commit** changes (if any) with `[skip ci]` tag
4. **Check** for remaining errors

### What This Means

- Push your code without worrying about formatting
- CI will auto-fix most issues and push them back
- Only critical errors (that can't be auto-fixed) will fail CI
- Pull the auto-fixes with: `git pull`

---

## 💡 Tips

### Before Pushing

```bash
# Quick check
python scripts/lint_and_format.py --check

# Fix everything
python scripts/lint_and_format.py

# Commit fixes
git add -u
git commit -m "chore: apply linting fixes"
git push
```

### Disable Hook Temporarily

```bash
git push --no-verify  # Skip pre-push hook
```

### See What Changed

```bash
# After auto-formatting
git diff
```

---

## 🔧 Configuration

### Ruff Config (`pyproject.toml`)

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM"]
ignore = ["ARG002", "SIM117", "F541", "B905", ...]  # Relaxed rules

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### CI Workflow (`.github/workflows/ci.yml`)

The lint job now:
1. Auto-formats code
2. Auto-fixes linting issues
3. Commits and pushes fixes automatically
4. Only fails on unfixable errors

---

## 📚 Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [Pre-commit Hooks](https://pre-commit.com/)

---

## ❓ Troubleshooting

### Hook Not Running

```bash
# Make hook executable
chmod +x .git/hooks/pre-push

# Or on Windows
icacls .git\hooks\pre-push /grant Everyone:F
```

### CI Still Failing

```bash
# Pull auto-fixes from CI
git pull

# Run local check
python scripts/lint_and_format.py --check
```

### Conflicts from Auto-Commits

```bash
# Pull with rebase
git pull --rebase

# Or merge
git pull
```

---

## 🎉 Benefits

1. **No More Linting Battles**: CI auto-fixes most issues
2. **Consistent Code Style**: Automatic formatting ensures uniformity
3. **Fast Development**: No manual formatting needed
4. **Fewer Failed PRs**: Pre-push hook catches issues early
5. **Relaxed Rules**: Focus on what matters, ignore nitpicks

---

**Last Updated**: 2025-01-22

# Contributing to EarthTrend Explorer

Thank you for your interest in contributing to **EarthTrend Explorer** (NASA Space Apps Challenge 2026: *"Be An Earth System Trend Detective!"*)!

This document outlines the collaborative workflow, coding conventions, branching strategies, and testing requirements for all researchers and developers contributing to the project.

---

## 1. Code of Conduct & Scientific Integrity Principles

As a scientific data-analysis platform built on NASA Earth Science observations, all contributors must uphold the following non-negotiable principles:
1. **Zero Data Fabrication**: Never invent numbers, synthesize artificial measurements, or extrapolate trends without empirical evidence.
2. **Transparent Provenance**: Always cite the exact NASA dataset, model version (e.g. MERRA-2), spatial grid resolution, and retrieval timestamp.
3. **Epistemological Candor**: Clearly distinguish lack of statistical trend detection from evidence of climate stability. Never overstate statistical significance.
4. **Reproducibility**: Any new statistical method or data loader must be accompanied by automated unit tests.

---

## 2. Git Branching Strategy

To avoid merge conflicts when multiple researchers work simultaneously, we follow a feature-branch workflow rooted on `main`:

```
main (stable shared branch)
 │
 ├── feature/<feature-name>    (e.g., feature/slope-interaction-test)
 ├── fix/<bug-name>            (e.g., fix/alexandria-tie-correction)
 ├── research/<experiment>     (e.g., research/gpm-precipitation-eval)
 └── docs/<doc-topic>          (e.g., docs/expand-mathematical-derivations)
```

### Branch Rules
- **`main`**: Protected integration branch. All code on `main` must pass all automated tests (`pytest -v`).
- **Never push directly to `main`**: Always develop on a topic branch and create a Pull Request.
- **Isolate by Subsystem**:
  - Statistical modifications $\to$ `feature/stats-...`
  - Visualization additions $\to$ `feature/viz-...`
  - Data loader extensions $\to$ `feature/data-...`

---

## 3. Getting Started & Development Setup

1. **Fork or Clone**:
   ```bash
   git clone https://github.com/ammarsami255/Be-An-Earth-System-Trend-Detective.git
   cd Be-An-Earth-System-Trend-Detective
   ```

2. **Environment Setup**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Existing Tests**:
   ```bash
   # On Windows (PowerShell):
   $env:PYTHONIOENCODING="utf-8"

   # Run test suite:
   pytest -v
   ```
   Do not begin development until all existing 36 tests pass on your machine (100% pass rate).

---

## 4. Coding & Documentation Standards

- **Python Version**: Python 3.10+ compatible.
- **Style Guide**: Follow PEP 8 guidelines. Use meaningful variable names (`elapsed_years`, `s_statistic`, not `x1`, `tmp`).
- **Type Annotations**: All public functions in `src/` must include typing hints (`from typing import Tuple, Dict, Any, List, Optional`).
- **Docstrings**: Include comprehensive docstrings explaining inputs, outputs, mathematical formulation, and units.
- **Units Discipline**: Whenever returning rates, explicitly specify units (e.g., `units/year`, `units/decade`).

---

## 5. Testing Requirements

Every Pull Request must satisfy the following testing standards:
1. **Unit Tests**: Place in `tests/test_<subsystem>.py`.
2. **Deterministic Random Seeds**: Always set `np.random.seed(...)` in synthetic tests to prevent intermittent test failures.
3. **Analytical Verification**: Where applicable, test against known closed-form analytical solutions or reference libraries (`scipy.stats`, `pymannkendall`).
4. **AppTest UI Verification**: If adding or altering Streamlit UI elements, run the headless AppTest suite:
   ```bash
   pytest tests/test_streamlit_app.py -v
   ```
5. **Run Full Test Suite Before Committing**:
   ```bash
   pytest -v
   ```

---

## 6. Commit Message Conventions

Use clear, structured commit messages following Conventional Commits:

```
<type>(<scope>): <short description>

[optional detailed body explaining why the change was made]
```

### Types
- `feat`: A new feature or analytical capability.
- `fix`: A bug fix or mathematical correction.
- `docs`: Documentation additions or revisions.
- `test`: Adding or updating test suites.
- `refactor`: Code refactoring without behavioral or statistical changes.
- `chore`: Maintenance tasks (e.g., `.gitignore`, `requirements.txt`).

### Examples
- `feat(trend): add multivariate slope interaction test`
- `fix(loader): handle trailing commas in NASA POWER headers`
- `docs(math): add Gilbert confidence interval derivation`

---

## 7. Pull Request Workflow

1. Rebase your topic branch on the latest `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/<your-feature>
   git rebase main
   ```
2. Run `pytest -v` to confirm no regressions.
3. Push to your remote topic branch:
   ```bash
   git push origin feature/<your-feature>
   ```
4. Open a Pull Request on GitHub against `main`.
5. Fill out the PR template:
   - What problem does this solve?
   - What mathematical or technical approach was taken?
   - How was it tested?
   - Does this alter any existing statistical outputs?

---

## 8. Reporting Issues & Proposing Research

- **Bug Reports**: Open an issue describing the input dataset, expected behavior, observed behavior, and full traceback.
- **Statistical Questions**: Refer to [`MATHEMATICAL_AND_TECHNICAL_DECISIONS.md`](file:///D:/Space-Apps/MATHEMATICAL_AND_TECHNICAL_DECISIONS.md) before opening methodology inquiries.

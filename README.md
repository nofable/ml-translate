# ml-translate

A neural machine translation project using PyTorch.

## Installation

```bash
uv sync
```

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run tests with verbose output:
```bash
uv run pytest -v
```

Run tests excluding slow tests:
```bash
uv run pytest -m "not slow"
```

Run tests with coverage report:
```bash
uv run pytest --cov=ml_translate --cov-report=term-missing
```

Run a specific test file:
```bash
uv run pytest tests/test_data.py
```

Run a specific test class or function:
```bash
uv run pytest tests/test_data.py::TestLang
uv run pytest tests/test_data.py::TestLang::test_lang_init
```

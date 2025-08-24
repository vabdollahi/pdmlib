## Requirements

- Python 3.12+
- [`uv` package manager](https://github.com/pdm-project/uv)

### Common uv Commands

- **Install dependencies:**
  `uv sync --all-extras`
- **Add dependency:**
  `uv add <package_name>`
- **Add dev dependency:**
  `uv add --dev <package_name>`
- **Remove dependency:**
  `uv remove <package_name>`

Dependencies are managed in `pyproject.toml` and `uv.lock`.

### Running the Application

- **Dev server (auto-reload):**
  `uv run uvicorn app.main:app --reload`
- **Production server:**
  `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4`


## Development Scripts

Reusable scripts are defined in `pyproject.toml`. Run them with:
```
  uv run task <script>
```
Examples:
```
  uv run task start        # Runs the FastAPI server (dev)
  uv run task test         # Runs all tests
  uv run task lint         # Runs code linting checks
  uv run task lint-format  # Formats and sorts imports
```

## Testing

Tests are in the `tests/` directory. Run them with:
```
  uv run task test
```
Or using pytest directly:
```
  pytest tests/
```

## Linting & Formatting

Lint code:
```
  uv run task lint
```
Format code and sort imports:
```
  uv run task lint-format
```
Or run the tools directly:
```
  black app tests
  isort app tests
```
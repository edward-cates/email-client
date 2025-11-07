# email-client
An email client to classify my emails where everything runs locally.

## Development

### Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting.

**Install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

**Run linting:**
```bash
make lint
# or
ruff check src/
```

**Format code:**
```bash
make format
# or
ruff format src/
```

**Check formatting without making changes:**
```bash
make check
# or
ruff format --check src/
```

**Auto-fix linting issues:**
```bash
make fix
# or
ruff check --fix src/
ruff format src/
```

See `pyproject.toml` for linting configuration.

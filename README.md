# PDM Power Management Library

A comprehensive power management simulation library for solar + battery systems with real-time market integration.

## Requirements

- Python 3.12+
- [`uv` package manager](https://docs.astral.sh/uv/)

## Quick Start

```bash
# Install dependencies
uv sync

# Run solar revenue calculator (simple demo)
uv run task start

# Run power management simulation (full system)
uv run task simulate

# Run tests
uv run task test

# Lint code
uv run task lint
```

## Project Structure

- `app/main.py` - Solar revenue calculator demo
- `app/simulation_main.py` - Complete power management simulation
- `tests/config/` - Configuration files for testing
  - `test_config_simple.json` - Single plant configuration
  - `test_config_multi.json` - Multi-plant portfolio configuration

## Key Features

- **Unified Configuration System**: JSON-driven setup for complete simulations
- **Agent-Based Control**: Heuristic agents for battery management
- **Real Market Data**: Integration with CAISO and IESO price feeds
- **PV Modeling**: Advanced photovoltaic simulation using PVLib
- **Portfolio Management**: Multi-plant coordination and optimization
- **UTC Timezone Enforcement**: Consistent time handling throughout

## Development Scripts

All scripts are defined in `pyproject.toml`:

```bash
uv run task start        # Solar revenue demo
uv run task simulate     # Power management simulation  
uv run task test         # Run all tests
uv run task test-verbose # Verbose test output
uv run task lint         # Lint checks
uv run task lint-fix     # Auto-fix linting issues
```
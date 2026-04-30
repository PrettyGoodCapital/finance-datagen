# finance datagen

Standard financial data generation

[![Build Status](https://github.com/prettygoodcapital/finance-datagen/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/prettygoodcapital/finance-datagen/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/prettygoodcapital/finance-datagen/branch/main/graph/badge.svg)](https://codecov.io/gh/prettygoodcapital/finance-datagen)
[![License](https://img.shields.io/github/license/prettygoodcapital/finance-datagen)](https://github.com/prettygoodcapital/finance-datagen)
[![PyPI](https://img.shields.io/pypi/v/finance-datagen.svg)](https://pypi.python.org/pypi/finance-datagen)

## Overview

`finance-datagen` produces **synthetic** financial time series for
testing, demos, and benchmarking the rest of the `finance-*` stack
without relying on real market data. The numerical core is implemented
in Rust and emits Apache Arrow `RecordBatch` values; the Python layer
wraps each generator so the public API returns `polars.DataFrame`
objects.

### Generators

#### Price models (Rust core)

| Symbol            | Model                                                       | Output columns                                      |
| ----------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| `GBMGenerator`    | Geometric Brownian Motion (log-Euler)                       | `timestamp, symbol, price`                          |
| `HestonGenerator` | Heston (1993) stochastic volatility (full-truncation Euler) | `timestamp, symbol, price, variance`                |
| `GARCHGenerator`  | GARCH(1,1) returns                                          | `timestamp, symbol, price, return, sigma`           |
| `ohlc_from_close` | OHLCV synthesis from any close series                       | `timestamp, symbol, open, high, low, close, volume` |

#### Cross-sectional panels (Python)

| Symbol                     | Output                                                                       |
| -------------------------- | ---------------------------------------------------------------------------- |
| `generate_signal`          | Long-form `[date, symbol, signal, fwd_returns]` with target Pearson IC       |
| `generate_factor_loadings` | Wide `[symbol, market, value, momentum, size, quality]` Barra-style loadings |
| `generate_benchmark`       | `[date, benchmark]` Gaussian benchmark return series                         |

All Rust generators accept an optional `seed: int` for bit-reproducible
output across platforms (ChaCha8 RNG); the cross-sectional generators
accept a `seed` for `numpy.random.default_rng`.

### Quick start

```python
from finance_datagen import GBMGenerator, ohlc_from_close

closes = GBMGenerator(s0=100.0, mu=0.07, sigma=0.25, seed=0).generate()
bars   = ohlc_from_close(closes["price"], seed=0)
```

See the [Data](docs/src/DATA.md) page for model math, parameter ranges,
and output schemas, and the [API](docs/src/API.md) page for a complete
function-level reference.

### Architecture

The Rust core (`rust/src/`) is **polars-free**: every generator builds
an `arrow_array::RecordBatch` and returns it through the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
PyCapsule via `pyo3-arrow`. The Python wrappers call
`polars.from_arrow(batch)` on the receiving end. This keeps the
polars-rs and polars-py codebases on opposite sides of a stable ABI
boundary, avoiding the binary-incompatibility issues that come with
linking polars from both Rust and CPython.

> [!NOTE]
> This library was generated using [copier](https://copier.readthedocs.io/en/stable/) from the [Base Python Project Template repository](https://github.com/python-project-templates/base).

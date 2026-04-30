# API

`finance-datagen` exposes the following public symbols at the top of
the package:

```python
from finance_datagen import (
    # Rust-backed price-path generators
    GBMGenerator,
    HestonGenerator,
    GARCHGenerator,
    ohlc_from_close,
    # Python cross-sectional generators
    generate_signal,
    generate_factor_loadings,
    generate_benchmark,
)
```

Each price-path generator class follows the same pattern: instantiate
with parameters, then call `.generate()` to obtain a polars
`DataFrame`. The cross-sectional generators are plain functions that
return a `DataFrame` directly.

For the precise math, parameter ranges, and output schemas of every
model, see the [Data](DATA.md) page.

---

## Quick start

```python
import polars as pl
from finance_datagen import GBMGenerator, ohlc_from_close

# 1 year of daily log-normal closes, deterministic.
prices = GBMGenerator(
    s0=100.0,
    mu=0.07,
    sigma=0.25,
    n_steps=252,
    symbol="ACME",
    seed=0,
).generate()

# Synthesize OHLCV bars around the closes.
bars = ohlc_from_close(prices["price"], symbol="ACME", seed=0)

print(bars.head())
```

---

## Reference

```{eval-rst}
.. currentmodule:: finance_datagen

.. autoclass:: GBMGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: HestonGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: GARCHGenerator
   :members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: ohlc_from_close

.. autofunction:: generate_signal

.. autofunction:: generate_factor_loadings

.. autofunction:: generate_benchmark
```

## Recipes

### Composing models

`ohlc_from_close` is generator-agnostic — feed it any close series.

```python
from finance_datagen import HestonGenerator, ohlc_from_close

px = HestonGenerator(seed=42).generate()
bars = ohlc_from_close(px["price"], symbol="HEST", seed=42)
```

### Long horizons

```python
# 10 years of daily data
GBMGenerator(n_steps=252 * 10, seed=0).generate()
```

### Custom timestamp grids

`step_ms` and `start_ms` are independent of `dt`, so you can produce a
high-frequency timestamp grid for visual inspection while keeping the
SDE on a daily scale:

```python
GBMGenerator(
    n_steps=1000,
    dt=1/252,                # daily-scale variance
    start_ms=1_700_000_000_000,
    step_ms=60_000,          # 1-minute timestamps
    seed=0,
).generate()
```

### Bypassing the polars wrapper

If you need the raw `pyarrow.RecordBatch` (e.g. to write Parquet
without round-tripping through polars):

```python
from finance_datagen.finance_datagen import GBMGenerator as RustGBM
import pyarrow.parquet as pq
import pyarrow as pa

batch = RustGBM(seed=0).record_batch()
pq.write_table(pa.Table.from_batches([batch]), "gbm.parquet")
```

---

## Versioning

The current version is exposed at `finance_datagen.__version__`. The
public API is the symbols listed above; private members
(`_inner`, `_rb_to_polars`, `finance_datagen.finance_datagen.*`) are
not part of the SemVer contract and may change between releases.

# API

`finance-datagen` exposes the following public symbols at the top of
the package:

```python
from finance_datagen import (
    DataGenerator,
    # Rust-backed price-path generators
    GBMGenerator,
    HestonGenerator,
    GARCHGenerator,
    ohlc_from_close,
    generate_prices,
    generate_gbm,
    generate_heston,
    generate_garch,
    # Python cross-sectional generators
    SignalGenerator,
    FactorLoadingsGenerator,
    BenchmarkGenerator,
    generate_signal,
    generate_factor_loadings,
    generate_benchmark,
    # Python portfolio, market, and risk-model generators
    PositionsGenerator,
    TransactionsGenerator,
    OrdersGenerator,
    ExecutionsGenerator,
    MultiAssetGBMGenerator,
    RegimeSwitchingGenerator,
    MarketImpactCurveGenerator,
    StatisticalRiskModelGenerator,
    FundamentalRiskModelGenerator,
    FactorCovarianceGenerator,
    SpecificVarianceGenerator,
    generate_positions,
    generate_transactions,
    generate_orders,
    generate_executions,
    generate_multi_asset_gbm,
    generate_regime_switching,
    generate_market_impact_curve,
    generate_statistical_risk_model,
    generate_fundamental_risk_model,
    generate_factor_covariance,
    generate_specific_variance,
)
```

Each generator class inherits from `DataGenerator`, a pydantic base
model. Instantiate with typed parameters, then call `.generate()` to
obtain the synthetic output. `next(generator)` is also supported as a
one-shot iterator convenience. The `generate_*` functions are thin
wrappers that instantiate the matching model for validation and return
`.generate()`.

For the precise math, parameter ranges, and output schemas of every
model, see the [Data](DATA.md) page.

______________________________________________________________________

## Quick start

```python
from finance_datagen import generate_prices, ohlc_from_close

# 1 year of daily log-normal closes, deterministic.
prices = generate_prices(symbol="ACME", seed=0)

# Synthesize OHLCV bars around the closes.
bars = ohlc_from_close(prices["price"], symbol="ACME", seed=0)

print(bars.head())
```

______________________________________________________________________

## Reference

```{eval-rst}
.. currentmodule:: finance_datagen

.. autoclass:: DataGenerator
    :members: generate
    :show-inheritance:

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

.. autofunction:: generate_prices

.. autofunction:: generate_gbm

.. autofunction:: generate_heston

.. autofunction:: generate_garch

.. autoclass:: SignalGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: FactorLoadingsGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: BenchmarkGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autofunction:: generate_signal

.. autofunction:: generate_factor_loadings

.. autofunction:: generate_benchmark

.. autoclass:: PositionsGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: TransactionsGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: OrdersGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: ExecutionsGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: MultiAssetGBMGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: RegimeSwitchingGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: MarketImpactCurveGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: StatisticalRiskModelGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: FundamentalRiskModelGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: FactorCovarianceGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autoclass:: SpecificVarianceGenerator
    :members:
    :special-members: __init__
    :show-inheritance:

.. autofunction:: generate_positions

.. autofunction:: generate_transactions

.. autofunction:: generate_orders

.. autofunction:: generate_executions

.. autofunction:: generate_multi_asset_gbm

.. autofunction:: generate_regime_switching

.. autofunction:: generate_market_impact_curve

.. autofunction:: generate_statistical_risk_model

.. autofunction:: generate_fundamental_risk_model

.. autofunction:: generate_factor_covariance

.. autofunction:: generate_specific_variance
```

### Post-trade fixtures

```python
from finance_datagen import (
    ExecutionsGenerator,
    OrdersGenerator,
    PositionsGenerator,
    TransactionsGenerator,
    generate_executions,
    generate_orders,
    generate_positions,
    generate_transactions,
)

positions = PositionsGenerator(n_dates=20, n_assets=50, seed=0).generate()
transactions = TransactionsGenerator(n_dates=20, n_assets=50, seed=0).generate()
orders = OrdersGenerator(n_dates=20, n_assets=50, seed=0).generate()
executions = ExecutionsGenerator(n_dates=20, n_assets=50, seed=0).generate()

# Equivalent convenience wrappers are available:
positions = generate_positions(n_dates=20, n_assets=50, seed=0)
transactions = generate_transactions(n_dates=20, n_assets=50, seed=0)
orders = generate_orders(n_dates=20, n_assets=50, seed=0)
executions = generate_executions(n_dates=20, n_assets=50, seed=0)
```

### Risk-model fixtures

```python
from finance_datagen import StatisticalRiskModelGenerator

model = StatisticalRiskModelGenerator(n_dates=252, n_assets=100, n_factors=5, seed=0).generate()
factor_loadings = model["factor_loadings"]
factor_returns = model["factor_returns"]
specific_variance = model["specific_variance"]
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

______________________________________________________________________

## Versioning

The current version is exposed at `finance_datagen.__version__`. The
public API is the symbols listed above; private members
(`_inner`, `_rb_to_polars`, `finance_datagen.finance_datagen.*`) are
not part of the SemVer contract and may change between releases.

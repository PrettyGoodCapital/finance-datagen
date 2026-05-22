# Data

`finance-datagen` produces **synthetic** financial time series. Every
Rust generator emits an [Apache Arrow](https://arrow.apache.org/)
`RecordBatch` from a polars-free Rust core, which the Python layer
wraps into a `polars.DataFrame` via the pyarrow PyCapsule interface.
Python-only generators return polars `DataFrame` objects directly.
All public generator classes inherit from `DataGenerator`, a pydantic
base model that validates typed parameters at construction time. Each
generator can be run with `.generate()` or with one-shot iterator syntax
via `next(generator)`.

This page documents the data: stochastic models, generator parameters,
and output schemas.

---

## Conventions

All tabular outputs share the following conventions:

| Aspect | Convention |
| --- | --- |
| Timestamp column | `timestamp` of type `Timestamp(Millisecond, UTC)` unless noted otherwise |
| Symbol column | `symbol` of type `Utf8` |
| Numeric columns | `Float64` |
| Path length | A path generator with `n_steps` returns `n_steps + 1` rows |
| Time grid | Uniform: `start_ms + i * step_ms` for `i = 0..=n_steps` |
| Reproducibility | Fixed `seed: int` gives deterministic outputs for that generator family |

Most generators also have a matching `generate_*` convenience function
that instantiates the model for validation and returns `.generate()`.

The `dt` parameter controls the modeling time step used in SDE
discretization; `step_ms` controls only the timestamp column. They are
independent: a daily model (`dt = 1/252`) can be emitted on a second or
minute timestamp grid for testing.

---

## Models

### Geometric Brownian Motion (GBM)

The classic Black-Scholes log-normal price process.

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

Discretized exactly in log-space:

$$
S_{t+1} = S_t \exp\!\Big( (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\sqrt{dt}\,Z \Big),
\quad Z \sim \mathcal{N}(0, 1)
$$

#### GBM Parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `s0` | `100.0` | initial price, must be > 0 |
| `mu` | `0.05` | drift, annualized |
| `sigma` | `0.2` | volatility, annualized and nonnegative |
| `dt` | `1/252` | model time step in years |
| `n_steps` | `252` | number of return draws |
| `symbol` | `"SYM"` | label written into the `symbol` column |
| `start_ms` | `0` | first timestamp, epoch ms UTC |
| `step_ms` | `86_400_000` | timestamp spacing, one day by default |
| `seed` | `None` | RNG seed |

#### GBM Schema

| Column | Type | Notes |
| --- | --- | --- |
| `timestamp` | `Timestamp(ms, UTC)` | uniform grid |
| `symbol` | `Utf8` | constant |
| `price` | `Float64` | strictly positive |

---

### Heston Stochastic Volatility

Two-factor SDE with mean-reverting variance and correlated price and
variance shocks:

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{S} \\
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{v} \\
\mathrm{Corr}(dW^S, dW^v) &= \rho
\end{aligned}
$$

The implementation uses full-truncation Euler on variance, then
log-Euler on price:

$$
\begin{aligned}
v_{t+1} &= v_t + \kappa(\theta - v_t^+)\,dt + \xi\sqrt{v_t^+\,dt}\,Z_v \\
S_{t+1} &= S_t \exp\!\Big( (\mu - \tfrac{1}{2}v_t^+)\,dt + \sqrt{v_t^+\,dt}\,Z_S \Big)
\end{aligned}
$$

where $v_t^+ = \max(v_t, 0)$.

#### Heston Parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `s0` | `100.0` | initial price > 0 |
| `v0` | `0.04` | initial variance >= 0 |
| `mu` | `0.05` | risk-neutral or physical drift |
| `kappa` | `2.0` | mean-reversion speed, nonnegative |
| `theta` | `0.04` | long-run variance, nonnegative |
| `xi` | `0.3` | vol-of-vol, nonnegative |
| `rho` | `-0.7` | leverage correlation, must satisfy `abs(rho) <= 1` |
| `dt` | `1/252` | model time step in years |
| `n_steps` | `252` | number of draws |

#### Heston Schema

| Column | Type | Notes |
| --- | --- | --- |
| `timestamp` | `Timestamp(ms, UTC)` | uniform grid |
| `symbol` | `Utf8` | constant |
| `price` | `Float64` | strictly positive |
| `variance` | `Float64` | nonnegative after truncation |

---

### GARCH(1,1) Returns

Discrete-time conditional-variance model in log returns:

$$
\begin{aligned}
r_t &= \mu + \sigma_t Z_t,\quad Z_t \sim \mathcal{N}(0,1) \\
\sigma_t^2 &= \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2 \\
\varepsilon_{t-1} &= r_{t-1} - \mu
\end{aligned}
$$

When `alpha + beta < 1`, the initial variance is the unconditional
variance `omega / (1 - alpha - beta)`. Otherwise it falls back to
`omega`.

#### GARCH Parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `s0` | `100.0` | initial price > 0 |
| `mu` | `0.0` | mean log return |
| `omega` | `1e-6` | constant variance term >= 0 |
| `alpha` | `0.05` | shock weight >= 0 |
| `beta` | `0.90` | persistence weight >= 0 |
| `n_steps` | `252` | number of return draws |

#### GARCH Schema

| Column | Type | Notes |
| --- | --- | --- |
| `timestamp` | `Timestamp(ms, UTC)` | uniform grid |
| `symbol` | `Utf8` | constant |
| `price` | `Float64` | updated with `S_t = S_{t-1} exp(r_t)` |
| `return` | `Float64` | first row is `0.0` |
| `sigma` | `Float64` | conditional volatility, strictly positive |

---

### OHLCV Synthesis From Close

`ohlc_from_close()` takes any close series and synthesizes plausible
Open/High/Low/Volume columns around it.

For each bar `i`:

```text
open_i  = close_{i-1}                          (open_0 = close_0)
high_i  = max(open_i, close_i) * (1 + |U_1| * intrabar_vol)
low_i   = min(open_i, close_i) * (1 - |U_2| * intrabar_vol)
ret_i   = log(close_i / close_{i-1})           (ret_0 = 0)
vol_i   = base_volume + vol_factor * |ret_i|
```

This guarantees `high >= max(open, close)` and
`low <= min(open, close)`.

#### OHLCV Parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `close` | required | iterable, numpy array, or `pl.Series` of floats |
| `intrabar_vol` | `0.005` | per-bar high/low envelope width |
| `base_volume` | `1_000_000` | floor volume |
| `vol_factor` | `5e7` | volume sensitivity to absolute log return |
| `symbol` | `"SYM"` | symbol label |
| `start_ms` | `0` | first timestamp, epoch ms UTC |
| `step_ms` | `86_400_000` | timestamp spacing |
| `seed` | `None` | RNG seed |

#### OHLCV Schema

| Column | Type |
| --- | --- |
| `timestamp` | `Timestamp(ms, UTC)` |
| `symbol` | `Utf8` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |
| `volume` | `Float64` |

---

## Cross-Sectional Panels

Three pydantic generator models produce panels for testing alpha and
risk code that operates over `(date, symbol)` pairs. The legacy
`generate_signal`, `generate_factor_loadings`, and `generate_benchmark`
helpers remain as thin wrappers around the matching model classes. They
use `numpy.random.default_rng(seed)`.

### `SignalGenerator`

Long-form panel `[date, symbol, signal, fwd_returns]` constructed so
that the cross-sectional Pearson IC of `signal` against `fwd_returns`
is approximately `ic` per date:

$$
\mathrm{signal} = ic \cdot z(\mathrm{fwd}) + \sqrt{1 - ic^2}\,\varepsilon
$$

| Param | Default | Meaning |
| --- | --- | --- |
| `n_dates` | `252` | rows in the date dimension |
| `n_assets` | `50` | rows in the asset dimension |
| `ic` | `0.05` | target per-date Pearson IC, in `(-1, 1)` |
| `return_vol` | `0.02` | cross-sectional standard deviation of fwd returns |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first date |
| `symbols` | `None` | optional explicit symbol list |

Output schema: `date`, `symbol`, `signal`, `fwd_returns`.

Convenience wrapper: `generate_signal(...)`.

### `FactorLoadingsGenerator`

Wide-form Barra-style factor loadings, one row per asset. The `market`
factor is set to 1.0 when present. Other factors are drawn from a
standard normal distribution and standardized cross-sectionally.

| Param | Default | Meaning |
| --- | --- | --- |
| `n_assets` | `50` | row count |
| `factors` | `("market", "value", "momentum", "size", "quality")` | factor columns |
| `seed` | `None` | numpy RNG seed |
| `symbols` | `None` | optional explicit symbol list |

Output schema: `symbol` plus one numeric column per factor.

Convenience wrapper: `generate_factor_loadings(...)`.

### `BenchmarkGenerator`

Independent Gaussian benchmark return series with target annualized mean
and volatility.

| Param | Default | Meaning |
| --- | --- | --- |
| `n_dates` | `252` | row count |
| `annual_return` | `0.08` | target annualized mean |
| `annual_vol` | `0.16` | target annualized volatility |
| `periods_per_year` | `252` | annualization factor |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first date |

Output schema: `date`, `benchmark`.

Convenience wrapper: `generate_benchmark(...)`.

---

## Portfolio and Transaction Generators

These Python generators create deterministic post-trade fixtures for
turnover, transaction cost, and execution quality tests.

### `PositionsGenerator`

Long-form position panel with one row per `(date, symbol)`. Per-date
absolute weights are normalized to `gross_exposure`; `market_value` is
`weight * portfolio_value`, and `quantity` is `market_value / price`.

| Param | Default | Meaning |
| --- | --- | --- |
| `n_dates` | `252` | number of dates |
| `n_assets` | `50` | assets per date |
| `portfolio_value` | `1_000_000.0` | equity denominator for market value |
| `gross_exposure` | `1.0` | per-date sum of absolute weights |
| `average_price` | `100.0` | center of synthetic price distribution |
| `price_vol` | `0.02` | daily log-return volatility for marks |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first date |
| `symbols` | `None` | optional explicit symbol list |

Output schema: `date`, `symbol`, `price`, `quantity`, `market_value`,
`weight`.

Convenience wrapper: `generate_positions(...)`.

### `TransactionsGenerator`

Synthetic transaction log with enum-backed `side` and `position_effect`
labels from `finance-enums`. `amount` is positive for `Buy` and
negative for `Sell`; opening and closing intent is represented by
`position_effect`. `notional` is `abs(amount) * price`; `fees` are
computed from `fee_bps`.

| Param | Default | Meaning |
| --- | --- | --- |
| `n_dates` | `252` | number of trade dates |
| `n_assets` | `50` | symbol universe size |
| `trades_per_day` | `25` | rows per date |
| `average_price` | `100.0` | center of trade-price distribution |
| `price_vol` | `0.25` | lognormal price dispersion |
| `max_amount` | `1_000` | max absolute share amount per row |
| `commission` | `1.0` | explicit commission per trade |
| `fee_bps` | `0.2` | explicit fee rate in basis points |
| `bps` | `5.0` | slippage or cost assumption column |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first trade date |
| `symbols` | `None` | optional explicit symbol list |

Output schema: `timestamp`, `symbol`, `amount`, `price`, `side`,
`position_effect`, `notional`, `commission`, `fees`, `bps`.

Convenience wrapper: `generate_transactions(...)`.

---

## Multi-Asset, Regime, and Market-Impact Generators

### `MultiAssetGBMGenerator`

Correlated multi-asset GBM with either a constant off-diagonal
correlation `rho` or a caller-provided correlation matrix `corr`. The
output is long-form `[timestamp, symbol, price, return]`; the first row
for each symbol has `return = 0.0`.

Convenience wrapper: `generate_multi_asset_gbm(...)`.

### `RegimeSwitchingGenerator`

Single-symbol Markov regime-switching path. The transition matrix rows
must sum to one. Regime-specific means and volatilities generate log
returns, and the output includes an integer `regime` label per timestamp.

Convenience wrapper: `generate_regime_switching(...)`.

### `MarketImpactCurveGenerator`

Generates participation-rate curves for temporary and permanent market
impact. The default model uses square-root temporary impact and linear
permanent impact:

```text
temporary_impact_bps = temporary_impact_coef * volatility * sqrt(participation_rate) * 10_000
permanent_impact_bps = permanent_impact_coef * volatility * participation_rate * 10_000
```

Output schema: `symbol`, `participation_rate`, `adv`, `volatility`,
`temporary_impact_bps`, `permanent_impact_bps`, `total_impact_bps`.

Convenience wrapper: `generate_market_impact_curve(...)`.

---

## Risk-Model Generators

### `StatisticalRiskModelGenerator`

Creates a synthetic asset-return matrix from latent factors, then fits a
PCA-style statistical risk model. `.generate()` returns a dictionary
with three polars frames:

| Key | Schema |
| --- | --- |
| `factor_loadings` | `symbol`, `factor_1`, ..., `factor_n` |
| `factor_returns` | `date`, `factor_1`, ..., `factor_n` |
| `specific_variance` | `symbol`, `specific_variance` |

Convenience wrapper: `generate_statistical_risk_model(...)`.

### `FundamentalRiskModelGenerator`

Creates Barra-style factor loadings with a categorical `sector` drawn
from the `finance-enums` sector taxonomy, a constant `market` exposure
of `1.0`, standardized style factors
(`value`, `momentum`, `size`, `quality`, `low_vol`, `growth` by
default), and positive `specific_variance`.

Convenience wrapper: `generate_fundamental_risk_model(...)`.

### `FactorCovarianceGenerator`

Creates a symmetric positive semidefinite covariance matrix with a
leading `factor` label column and one numeric column per factor. Factor
volatilities decay by `eigen_decay`, and cross-factor correlations decay
with factor distance.

Convenience wrapper: `generate_factor_covariance(...)`.

### `SpecificVarianceGenerator`

Creates a positive idiosyncratic variance vector with lognormal
dispersion around `target_vol ** 2`.

Convenience wrapper: `generate_specific_variance(...)`.

---

## Reproducibility

Every generator and `ohlc_from_close` accept an optional `seed: int`.
Rust generators initialize a `ChaCha8` PRNG (via `rand_chacha`), which
is portable across platforms and architectures. Python generators
initialize `numpy.random.default_rng(seed)` and are deterministic within
the same numpy version.

```python
from finance_datagen import GBMGenerator

a = GBMGenerator(seed=42).generate()
b = GBMGenerator(seed=42).generate()
assert a.equals(b)
```

If `seed` is omitted, the generator seeds from OS entropy and the path
will differ on every call.

---

## Why Arrow?

The Rust core never imports polars. Polars-rs and the polars Python
wheel use incompatible internal ABIs, so linking polars on both sides
of the FFI boundary leads to crashes that are extremely hard to debug.
Arrow is a stable, language-agnostic columnar format: the Rust side
builds an `arrow_array::RecordBatch`, hands it to Python over the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
PyCapsule, and the Python side calls `polars.from_arrow(batch)` to wrap
the same buffers into a `polars.DataFrame`.

If you prefer to skip the polars wrapping, you can pull the raw
`pyarrow.RecordBatch` out of the Rust extension directly:

```python
from finance_datagen.finance_datagen import GBMGenerator as RustGBM

batch = RustGBM(seed=0).record_batch()  # pyarrow.RecordBatch
```

# Data

`finance-datagen` produces **synthetic** financial time series. Every
generator emits an [Apache Arrow](https://arrow.apache.org/)
`RecordBatch` from a polars-free Rust core, which the Python layer
wraps into a `polars.DataFrame` via the pyarrow PyCapsule interface.

This page documents the data — the stochastic models, their parameters,
and the schema of every output table.

---

## Conventions

All tabular outputs share the following conventions:

| Aspect | Convention |
|---|---|
| Timestamp column | `timestamp` of type `Timestamp(Millisecond, UTC)` |
| Symbol column | `symbol` of type `Utf8` (constant per generator instance) |
| Numeric columns | `Float64` |
| Path length | A generator with `n_steps` returns `n_steps + 1` rows (initial state + drawn) |
| Time grid | Uniform: `start_ms + i * step_ms` for `i = 0..=n_steps` |
| Reproducibility | Fixed `seed: int` → identical paths (ChaCha8 RNG) |

The `dt` parameter (in years) controls the *modeling* time step used in
the SDE discretization; `step_ms` controls only the timestamp column.
They are independent — you can run a daily-frequency model
(`dt = 1/252`) on a wall-clock grid of seconds (`step_ms = 1000`) for
testing.

---

## Models

### Geometric Brownian Motion (GBM)

The classic Black–Scholes log-normal price process.

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

Discretized exactly in log-space (no bias for any `dt`):

$$
S_{t+1} = S_t \exp\!\Big( (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\sqrt{dt}\,Z \Big),
\quad Z \sim \mathcal{N}(0, 1)
$$

**Parameters** (`GBMGenerator`)

| Param | Default | Meaning |
|---|---|---|
| `s0` | `100.0` | initial price, must be > 0 |
| `mu` | `0.05` | drift (annualized) |
| `sigma` | `0.2` | volatility (annualized), must be ≥ 0 |
| `dt` | `1/252` | model time step (years) |
| `n_steps` | `252` | number of return draws |
| `symbol` | `"SYM"` | label written into the `symbol` column |
| `start_ms` | `0` | first timestamp (epoch ms, UTC) |
| `step_ms` | `86_400_000` | timestamp grid spacing (1 day) |
| `seed` | `None` | RNG seed for reproducibility |

**Schema**

| Column | Type | Notes |
|---|---|---|
| `timestamp` | `Timestamp(ms, UTC)` | uniform grid |
| `symbol` | `Utf8` | constant |
| `price` | `Float64` | strictly positive |

---

### Heston (1993) stochastic volatility

Two-factor SDE with mean-reverting variance and a correlated price
process:

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{S} \\
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{v} \\
\mathrm{Corr}(dW^S, dW^v) &= \rho
\end{aligned}
$$

Discretized with **full-truncation Euler** on the variance — the
standard remedy for the well-known negativity bug of plain Euler when
the Feller condition $2\kappa\theta \ge \xi^2$ is violated:

$$
\begin{aligned}
v_{t+1} &= v_t + \kappa(\theta - v_t^+)\,dt + \xi\sqrt{v_t^+\,dt}\,Z_v \\
S_{t+1} &= S_t \exp\!\Big( (\mu - \tfrac{1}{2}v_t^+)\,dt + \sqrt{v_t^+\,dt}\,Z_S \Big)
\end{aligned}
$$

where $v_t^+ = \max(v_t, 0)$ and the correlated normals
$(Z_S, Z_v)$ are produced via a 2×2 Cholesky factor of the correlation
matrix with off-diagonal $\rho$.

**Parameters** (`HestonGenerator`)

| Param | Default | Meaning |
|---|---|---|
| `s0` | `100.0` | initial price > 0 |
| `v0` | `0.04` | initial variance ≥ 0 |
| `mu` | `0.05` | risk-neutral / physical drift |
| `kappa` | `2.0` | mean-reversion speed (κ ≥ 0) |
| `theta` | `0.04` | long-run variance (θ ≥ 0) |
| `xi` | `0.3` | vol-of-vol (ξ ≥ 0) |
| `rho` | `-0.7` | leverage correlation, must satisfy `|rho| ≤ 1` |
| `dt` | `1/252` | time step (years) |
| `n_steps` | `252` | number of draws |

**Schema**

| Column | Type | Notes |
|---|---|---|
| `timestamp` | `Timestamp(ms, UTC)` | |
| `symbol` | `Utf8` | |
| `price` | `Float64` | strictly positive |
| `variance` | `Float64` | non-negative (post-truncation) |

---

### GARCH(1,1) returns

Discrete-time conditional-variance model in log-returns:

$$
\begin{aligned}
r_t &= \mu + \sigma_t\,Z_t,\quad Z_t \sim \mathcal{N}(0,1) \\
\sigma_t^2 &= \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2 \\
\varepsilon_{t-1} &= r_{t-1} - \mu
\end{aligned}
$$

When the process is stationary ($\alpha + \beta < 1$), $\sigma_0^2$ is
initialized to the unconditional variance
$\bar{\sigma}^2 = \omega / (1 - \alpha - \beta)$. Otherwise it falls
back to $\omega$.

**Parameters** (`GARCHGenerator`)

| Param | Default | Meaning |
|---|---|---|
| `s0` | `100.0` | initial price > 0 |
| `mu` | `0.0` | mean log-return |
| `omega` | `1e-6` | constant variance term ≥ 0 |
| `alpha` | `0.05` | shock weight ≥ 0 |
| `beta` | `0.90` | persistence weight ≥ 0 |
| `n_steps` | `252` | number of return draws |

**Schema**

| Column | Type | Notes |
|---|---|---|
| `timestamp` | `Timestamp(ms, UTC)` | |
| `symbol` | `Utf8` | |
| `price` | `Float64` | $S_t = S_{t-1} e^{r_t}$ |
| `return` | `Float64` | first row is `0.0` |
| `sigma` | `Float64` | conditional volatility, strictly positive |

---

### OHLCV synthesis from a close series

`ohlc_from_close()` is a utility, not a stochastic model: it takes any
existing series of close prices (typically the `price` column from one
of the generators above) and synthesizes plausible Open/High/Low/Volume
columns around it.

For each bar `i`:

```text
open_i  = close_{i-1}                          (open_0 = close_0)
high_i  = max(open_i, close_i) * (1 + |U_1| * intrabar_vol)
low_i   = min(open_i, close_i) * (1 - |U_2| * intrabar_vol)
ret_i   = log(close_i / close_{i-1})           (ret_0 = 0)
vol_i   = base_volume + vol_factor * |ret_i|
```

with $U_1, U_2 \sim \mathcal{U}(0, 1)$ independent. This guarantees the
canonical bar invariants `high ≥ max(open, close)` and
`low ≤ min(open, close)` regardless of the `intrabar_vol` setting.

**Parameters**

| Param | Default | Meaning |
|---|---|---|
| `close` | (required) | iterable / list / numpy / `pl.Series` of floats |
| `intrabar_vol` | `0.005` | per-bar high/low envelope width |
| `base_volume` | `1_000_000` | floor volume |
| `vol_factor` | `5e7` | volume sensitivity to absolute log-return |
| `symbol` | `"SYM"` | |
| `start_ms` | `0` | |
| `step_ms` | `86_400_000` | |
| `seed` | `None` | RNG seed |

**Schema**

| Column | Type |
|---|---|
| `timestamp` | `Timestamp(ms, UTC)` |
| `symbol` | `Utf8` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |
| `volume` | `Float64` |

---

## Cross-sectional panels

Three pure-Python helpers produce panels for testing alpha / risk
calculations that operate over `(date, symbol)` pairs. They use
`numpy.random.default_rng(seed)` rather than the Rust ChaCha8 RNG;
seeded outputs are reproducible within a single numpy version.

### `generate_signal`

Long-form panel `[date, symbol, signal, fwd_returns]` constructed so
that the cross-sectional Pearson IC of `signal` against `fwd_returns`
is approximately `ic` per date. The signal is built from the
standardised forward returns plus orthogonal noise:

$$
\mathrm{signal} = ic \cdot z(\mathrm{fwd}) + \sqrt{1 - ic^2}\,\varepsilon
$$

| Param | Default | Meaning |
|---|---|---|
| `n_dates` | `252` | rows in the date dimension |
| `n_assets` | `50` | rows in the asset dimension |
| `ic` | `0.05` | target per-date Pearson IC, in `(-1, 1)` |
| `return_vol` | `0.02` | cross-sectional std of forward returns |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first date |
| `symbols` | `None` | optional explicit symbol list |

Output schema:

| Column | Type |
|---|---|
| `date` | `Date` |
| `symbol` | `Utf8` |
| `signal` | `Float64` |
| `fwd_returns` | `Float64` |

### `generate_factor_loadings`

Wide-form Barra-style factor loadings, one row per asset.

The `market` factor (when present) is set to 1.0 for every asset.
Every other factor is drawn iid $\mathcal{N}(0, 1)$ and standardised
cross-sectionally to mean zero, unit std (population, `ddof=0`).

| Param | Default | Meaning |
|---|---|---|
| `n_assets` | `50` | row count |
| `factors` | `("market", "value", "momentum", "size", "quality")` | column names |
| `seed` | `None` | numpy RNG seed |
| `symbols` | `None` | optional explicit symbol list |

Output schema: `symbol` (`Utf8`) plus one `Float64` column per factor.

### `generate_benchmark`

Independent Gaussian benchmark return series with target annualised
mean `annual_return` and target annualised volatility `annual_vol`.

| Param | Default | Meaning |
|---|---|---|
| `n_dates` | `252` | row count |
| `annual_return` | `0.08` | target annualised mean |
| `annual_vol` | `0.16` | target annualised volatility |
| `periods_per_year` | `252` | annualisation factor |
| `seed` | `None` | numpy RNG seed |
| `start` | `2020-01-01` | first date |

Output schema:

| Column | Type |
|---|---|
| `date` | `Date` |
| `benchmark` | `Float64` |

---

## Reproducibility

Every generator and `ohlc_from_close` accept an optional `seed: int`.
Internally the seed initializes a `ChaCha8` PRNG (via `rand_chacha`),
which is portable across platforms and architectures: the same seed
will produce bit-identical outputs on Linux/macOS/Windows and on
x86-64/aarch64.

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
the same buffers (zero-copy) into a `polars.DataFrame`.

If you prefer to skip the polars wrapping, you can pull the raw
`pyarrow.RecordBatch` out of the Rust extension directly:

```python
from finance_datagen.finance_datagen import GBMGenerator as RustGBM

batch = RustGBM(seed=0).record_batch()  # pyarrow.RecordBatch
```

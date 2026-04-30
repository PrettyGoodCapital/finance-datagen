//! pyo3 bindings for finance-datagen.
//!
//! Each `#[pyclass]` mirrors a Rust generator config + builder. The
//! `record_batch` method invokes the pure-Rust core, then hands the
//! resulting `arrow_array::RecordBatch` to Python via `pyo3-arrow`'s
//! `PyRecordBatch::to_pyarrow`. Python wraps that pyarrow `RecordBatch`
//! into a polars `DataFrame` on its side.

use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;

use ::finance_datagen::{
    ohlc_from_close as core_ohlc_from_close, GarchConfig as CoreGarchConfig,
    GarchGenerator as CoreGarch, GbmConfig as CoreGbmConfig, GbmGenerator as CoreGbm,
    HestonConfig as CoreHestonConfig, HestonGenerator as CoreHeston, OhlcConfig as CoreOhlcConfig,
};

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn rb_to_py<'py>(
    py: Python<'py>,
    rb: arrow_array::RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyRecordBatch::new(rb).into_pyarrow(py)?)
}

#[pyclass(module = "finance_datagen")]
pub struct GBMGenerator {
    cfg: CoreGbmConfig,
}

#[pymethods]
impl GBMGenerator {
    #[new]
    #[pyo3(signature = (
        s0 = 100.0,
        mu = 0.05,
        sigma = 0.2,
        dt = 1.0 / 252.0,
        n_steps = 252,
        symbol = "SYM".to_string(),
        start_ms = 0,
        step_ms = 86_400_000,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        s0: f64,
        mu: f64,
        sigma: f64,
        dt: f64,
        n_steps: usize,
        symbol: String,
        start_ms: i64,
        step_ms: i64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let cfg = CoreGbmConfig {
            s0,
            mu,
            sigma,
            dt,
            n_steps,
            symbol,
            start_ms,
            step_ms,
            seed,
        };
        // Validate eagerly.
        let _ = CoreGbm::new(cfg.clone()).map_err(map_err)?;
        Ok(Self { cfg })
    }

    /// Run the simulation and return a pyarrow `RecordBatch`.
    fn record_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rb = CoreGbm::new(self.cfg.clone())
            .map_err(map_err)?
            .record_batch()
            .map_err(map_err)?;
        rb_to_py(py, rb)
    }
}

#[pyclass(module = "finance_datagen")]
pub struct HestonGenerator {
    cfg: CoreHestonConfig,
}

#[pymethods]
impl HestonGenerator {
    #[new]
    #[pyo3(signature = (
        s0 = 100.0,
        v0 = 0.04,
        mu = 0.05,
        kappa = 2.0,
        theta = 0.04,
        xi = 0.3,
        rho = -0.7,
        dt = 1.0 / 252.0,
        n_steps = 252,
        symbol = "SYM".to_string(),
        start_ms = 0,
        step_ms = 86_400_000,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        s0: f64,
        v0: f64,
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        dt: f64,
        n_steps: usize,
        symbol: String,
        start_ms: i64,
        step_ms: i64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let cfg = CoreHestonConfig {
            s0,
            v0,
            mu,
            kappa,
            theta,
            xi,
            rho,
            dt,
            n_steps,
            symbol,
            start_ms,
            step_ms,
            seed,
        };
        let _ = CoreHeston::new(cfg.clone()).map_err(map_err)?;
        Ok(Self { cfg })
    }

    fn record_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rb = CoreHeston::new(self.cfg.clone())
            .map_err(map_err)?
            .record_batch()
            .map_err(map_err)?;
        rb_to_py(py, rb)
    }
}

#[pyclass(module = "finance_datagen")]
pub struct GARCHGenerator {
    cfg: CoreGarchConfig,
}

#[pymethods]
impl GARCHGenerator {
    #[new]
    #[pyo3(signature = (
        s0 = 100.0,
        mu = 0.0,
        omega = 1e-6,
        alpha = 0.05,
        beta = 0.90,
        n_steps = 252,
        symbol = "SYM".to_string(),
        start_ms = 0,
        step_ms = 86_400_000,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        s0: f64,
        mu: f64,
        omega: f64,
        alpha: f64,
        beta: f64,
        n_steps: usize,
        symbol: String,
        start_ms: i64,
        step_ms: i64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let cfg = CoreGarchConfig {
            s0,
            mu,
            omega,
            alpha,
            beta,
            n_steps,
            symbol,
            start_ms,
            step_ms,
            seed,
        };
        let _ = CoreGarch::new(cfg.clone()).map_err(map_err)?;
        Ok(Self { cfg })
    }

    fn record_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rb = CoreGarch::new(self.cfg.clone())
            .map_err(map_err)?
            .record_batch()
            .map_err(map_err)?;
        rb_to_py(py, rb)
    }
}

#[pyfunction]
#[pyo3(signature = (
    close,
    intrabar_vol = 0.005,
    base_volume = 1_000_000.0,
    vol_factor = 5e7,
    symbol = "SYM".to_string(),
    start_ms = 0,
    step_ms = 86_400_000,
    seed = None,
))]
#[allow(clippy::too_many_arguments)]
fn ohlc_from_close<'py>(
    py: Python<'py>,
    close: Vec<f64>,
    intrabar_vol: f64,
    base_volume: f64,
    vol_factor: f64,
    symbol: String,
    start_ms: i64,
    step_ms: i64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = CoreOhlcConfig {
        intrabar_vol,
        base_volume,
        vol_factor,
        symbol,
        start_ms,
        step_ms,
        seed,
    };
    let rb = core_ohlc_from_close(&close, &cfg).map_err(map_err)?;
    rb_to_py(py, rb)
}

#[pymodule]
fn finance_datagen(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<GBMGenerator>()?;
    m.add_class::<HestonGenerator>()?;
    m.add_class::<GARCHGenerator>()?;
    m.add_function(wrap_pyfunction!(ohlc_from_close, m)?)?;
    Ok(())
}

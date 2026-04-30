//! Geometric Brownian Motion price-path generator.
//!
//! Discrete exact solution:
//!
//! ```text
//! S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
//! ```
//!
//! where `Z ~ N(0, 1)`.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::arrow_util::{const_string, f64_array, record_batch, timestamp_grid_ms};
use crate::error::{DatagenError, DatagenResult};
use crate::rng::make_rng;
use crate::schema::price_schema;

#[derive(Clone, Debug)]
pub struct GbmConfig {
    /// Initial price `S_0`.
    pub s0: f64,
    /// Drift (annualized).
    pub mu: f64,
    /// Volatility (annualized).
    pub sigma: f64,
    /// Time step in years (e.g. `1.0 / 252.0` for daily).
    pub dt: f64,
    /// Number of steps to simulate.
    pub n_steps: usize,
    /// Symbol label.
    pub symbol: String,
    /// Start timestamp in milliseconds since epoch.
    pub start_ms: i64,
    /// Step in milliseconds between consecutive samples.
    pub step_ms: i64,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GbmConfig {
    fn default() -> Self {
        Self {
            s0: 100.0,
            mu: 0.05,
            sigma: 0.2,
            dt: 1.0 / 252.0,
            n_steps: 252,
            symbol: "SYM".into(),
            start_ms: 0,
            step_ms: 86_400_000,
            seed: None,
        }
    }
}

pub struct GbmGenerator {
    cfg: GbmConfig,
}

impl GbmGenerator {
    pub fn new(cfg: GbmConfig) -> DatagenResult<Self> {
        if cfg.s0 <= 0.0 {
            return Err(DatagenError::InvalidParameter("s0 must be > 0".into()));
        }
        if cfg.sigma < 0.0 {
            return Err(DatagenError::InvalidParameter("sigma must be >= 0".into()));
        }
        if cfg.dt <= 0.0 {
            return Err(DatagenError::InvalidParameter("dt must be > 0".into()));
        }
        Ok(Self { cfg })
    }

    /// Simulate the price path and return as `Vec<f64>`. Length is
    /// `n_steps + 1` (initial + n_steps drawn values).
    pub fn simulate(&self) -> Vec<f64> {
        let mut rng = make_rng(self.cfg.seed);
        let n = self.cfg.n_steps;
        let dt = self.cfg.dt;
        let drift = (self.cfg.mu - 0.5 * self.cfg.sigma * self.cfg.sigma) * dt;
        let diffusion = self.cfg.sigma * dt.sqrt();

        let mut prices = Vec::with_capacity(n + 1);
        let mut s = self.cfg.s0;
        prices.push(s);
        for _ in 0..n {
            let z: f64 = rng.sample(StandardNormal);
            s *= (drift + diffusion * z).exp();
            prices.push(s);
        }
        prices
    }

    /// Simulate and assemble into an Arrow `RecordBatch` with schema
    /// `price_schema()`.
    pub fn record_batch(&self) -> DatagenResult<RecordBatch> {
        let prices = self.simulate();
        let n = prices.len();
        let ts = timestamp_grid_ms(self.cfg.start_ms, self.cfg.step_ms, n);
        let sym = const_string(&self.cfg.symbol, n);
        let columns: Vec<ArrayRef> = vec![Arc::new(ts), Arc::new(sym), Arc::new(f64_array(prices))];
        record_batch(price_schema(), columns)
    }
}

/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let cfg = GbmConfig {
            seed: Some(42),
            n_steps: 10,
            ..GbmConfig::default()
        };
        let a = GbmGenerator::new(cfg.clone()).unwrap().simulate();
        let b = GbmGenerator::new(cfg).unwrap().simulate();
        assert_eq!(a, b);
        assert_eq!(a.len(), 11);
        assert!(a.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    #[test]
    fn rejects_bad_params() {
        assert!(GbmGenerator::new(GbmConfig {
            s0: 0.0,
            ..GbmConfig::default()
        })
        .is_err());
        assert!(GbmGenerator::new(GbmConfig {
            sigma: -1.0,
            ..GbmConfig::default()
        })
        .is_err());
        assert!(GbmGenerator::new(GbmConfig {
            dt: 0.0,
            ..GbmConfig::default()
        })
        .is_err());
    }

    #[test]
    fn record_batch_shape() {
        let g = GbmGenerator::new(GbmConfig {
            seed: Some(1),
            n_steps: 5,
            ..GbmConfig::default()
        })
        .unwrap();
        let rb = g.record_batch().unwrap();
        assert_eq!(rb.num_columns(), 3);
        assert_eq!(rb.num_rows(), 6);
    }
}

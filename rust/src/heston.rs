//! Heston (1993) stochastic-volatility price-path generator.
//!
//! Variance follows a CIR / square-root process with mean-reversion `kappa`
//! to `theta`, vol-of-vol `xi`, and price-variance correlation `rho`:
//!
//! ```text
//! dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
//! dS = mu * S * dt + sqrt(v) * S * dW1
//! corr(dW1, dW2) = rho
//! ```
//!
//! Implemented with a full-truncation Euler scheme on `v` (clamp at 0) and
//! the log-Euler scheme on `S` for stability.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::arrow_util::{const_string, f64_array, record_batch, timestamp_grid_ms};
use crate::error::{DatagenError, DatagenResult};
use crate::rng::make_rng;
use crate::schema::heston_schema;

#[derive(Clone, Debug)]
pub struct HestonConfig {
    pub s0: f64,
    pub v0: f64,
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub dt: f64,
    pub n_steps: usize,
    pub symbol: String,
    pub start_ms: i64,
    pub step_ms: i64,
    pub seed: Option<u64>,
}

impl Default for HestonConfig {
    fn default() -> Self {
        Self {
            s0: 100.0,
            v0: 0.04,
            mu: 0.05,
            kappa: 2.0,
            theta: 0.04,
            xi: 0.3,
            rho: -0.7,
            dt: 1.0 / 252.0,
            n_steps: 252,
            symbol: "SYM".into(),
            start_ms: 0,
            step_ms: 86_400_000,
            seed: None,
        }
    }
}

pub struct HestonGenerator {
    cfg: HestonConfig,
}

impl HestonGenerator {
    pub fn new(cfg: HestonConfig) -> DatagenResult<Self> {
        if cfg.s0 <= 0.0 {
            return Err(DatagenError::InvalidParameter("s0 must be > 0".into()));
        }
        if cfg.v0 < 0.0 {
            return Err(DatagenError::InvalidParameter("v0 must be >= 0".into()));
        }
        if cfg.kappa <= 0.0 || cfg.theta < 0.0 || cfg.xi < 0.0 {
            return Err(DatagenError::InvalidParameter(
                "kappa, theta, xi must be >= 0 (kappa > 0)".into(),
            ));
        }
        if !(-1.0..=1.0).contains(&cfg.rho) {
            return Err(DatagenError::InvalidParameter(
                "rho must be in [-1, 1]".into(),
            ));
        }
        if cfg.dt <= 0.0 {
            return Err(DatagenError::InvalidParameter("dt must be > 0".into()));
        }
        Ok(Self { cfg })
    }

    /// Simulate price and variance paths; returns `(prices, variances)`,
    /// each of length `n_steps + 1`.
    pub fn simulate(&self) -> (Vec<f64>, Vec<f64>) {
        let mut rng = make_rng(self.cfg.seed);
        let n = self.cfg.n_steps;
        let dt = self.cfg.dt;
        let sqrt_dt = dt.sqrt();
        let rho = self.cfg.rho;
        let sqrt_one_minus_rho2 = (1.0 - rho * rho).max(0.0).sqrt();

        let mut prices = Vec::with_capacity(n + 1);
        let mut variances = Vec::with_capacity(n + 1);
        let mut s = self.cfg.s0;
        let mut v = self.cfg.v0;
        prices.push(s);
        variances.push(v);

        for _ in 0..n {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            // Correlated normals
            let dw_s = z1;
            let dw_v = rho * z1 + sqrt_one_minus_rho2 * z2;

            let v_pos = v.max(0.0);
            let sqrt_v = v_pos.sqrt();

            // Variance: full-truncation Euler.
            v = v
                + self.cfg.kappa * (self.cfg.theta - v_pos) * dt
                + self.cfg.xi * sqrt_v * sqrt_dt * dw_v;
            v = v.max(0.0);

            // Log-Euler price update.
            let log_drift = (self.cfg.mu - 0.5 * v_pos) * dt;
            s *= (log_drift + sqrt_v * sqrt_dt * dw_s).exp();

            prices.push(s);
            variances.push(v);
        }
        (prices, variances)
    }

    pub fn record_batch(&self) -> DatagenResult<RecordBatch> {
        let (prices, variances) = self.simulate();
        let n = prices.len();
        let ts = timestamp_grid_ms(self.cfg.start_ms, self.cfg.step_ms, n);
        let sym = const_string(&self.cfg.symbol, n);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(ts),
            Arc::new(sym),
            Arc::new(f64_array(prices)),
            Arc::new(f64_array(variances)),
        ];
        record_batch(heston_schema(), columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let cfg = HestonConfig {
            seed: Some(7),
            n_steps: 50,
            ..HestonConfig::default()
        };
        let (a, av) = HestonGenerator::new(cfg.clone()).unwrap().simulate();
        let (b, bv) = HestonGenerator::new(cfg).unwrap().simulate();
        assert_eq!(a, b);
        assert_eq!(av, bv);
    }

    #[test]
    fn variance_nonnegative_and_prices_positive() {
        let g = HestonGenerator::new(HestonConfig {
            seed: Some(1),
            n_steps: 1000,
            ..HestonConfig::default()
        })
        .unwrap();
        let (p, v) = g.simulate();
        assert!(v.iter().all(|x| *x >= 0.0));
        assert!(p.iter().all(|x| *x > 0.0 && x.is_finite()));
    }

    #[test]
    fn rejects_bad_rho() {
        assert!(HestonGenerator::new(HestonConfig {
            rho: 1.5,
            ..HestonConfig::default()
        })
        .is_err());
    }
}

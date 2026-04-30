//! GARCH(1,1) discrete-time return generator.
//!
//! ```text
//! r_t = mu + sigma_t * z_t,    z_t ~ N(0, 1)
//! sigma_t^2 = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma_{t-1}^2
//! ```
//!
//! Stationarity requires `alpha + beta < 1`; we warn (do not enforce) if
//! the user passes a non-stationary parameter set.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::arrow_util::{const_string, f64_array, record_batch, timestamp_grid_ms};
use crate::error::{DatagenError, DatagenResult};
use crate::rng::make_rng;
use crate::schema::garch_schema;

#[derive(Clone, Debug)]
pub struct GarchConfig {
    pub s0: f64,
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub n_steps: usize,
    pub symbol: String,
    pub start_ms: i64,
    pub step_ms: i64,
    pub seed: Option<u64>,
}

impl Default for GarchConfig {
    fn default() -> Self {
        Self {
            s0: 100.0,
            mu: 0.0,
            omega: 1e-6,
            alpha: 0.05,
            beta: 0.90,
            n_steps: 252,
            symbol: "SYM".into(),
            start_ms: 0,
            step_ms: 86_400_000,
            seed: None,
        }
    }
}

pub struct GarchGenerator {
    cfg: GarchConfig,
}

impl GarchGenerator {
    pub fn new(cfg: GarchConfig) -> DatagenResult<Self> {
        if cfg.s0 <= 0.0 {
            return Err(DatagenError::InvalidParameter("s0 must be > 0".into()));
        }
        if cfg.omega < 0.0 || cfg.alpha < 0.0 || cfg.beta < 0.0 {
            return Err(DatagenError::InvalidParameter(
                "omega, alpha, beta must be >= 0".into(),
            ));
        }
        Ok(Self { cfg })
    }

    /// Returns `(prices, returns, sigmas)`, each of length `n_steps + 1`.
    /// The first row is the initial state with `return = 0`.
    pub fn simulate(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = make_rng(self.cfg.seed);
        let n = self.cfg.n_steps;

        // Unconditional variance (used as initial variance if stationary).
        let denom = 1.0 - self.cfg.alpha - self.cfg.beta;
        let var0 = if denom > 0.0 {
            self.cfg.omega / denom
        } else {
            self.cfg.omega.max(1e-12)
        };
        let mut sigma2 = var0;

        let mut prices = Vec::with_capacity(n + 1);
        let mut returns = Vec::with_capacity(n + 1);
        let mut sigmas = Vec::with_capacity(n + 1);
        let mut s = self.cfg.s0;
        prices.push(s);
        returns.push(0.0);
        sigmas.push(sigma2.sqrt());

        let mut prev_shock = 0.0_f64;
        for _ in 0..n {
            sigma2 =
                self.cfg.omega + self.cfg.alpha * prev_shock * prev_shock + self.cfg.beta * sigma2;
            let sigma = sigma2.sqrt();
            let z: f64 = rng.sample(StandardNormal);
            let shock = sigma * z;
            let r = self.cfg.mu + shock;
            prev_shock = shock;
            s *= (r).exp();
            prices.push(s);
            returns.push(r);
            sigmas.push(sigma);
        }
        (prices, returns, sigmas)
    }

    pub fn record_batch(&self) -> DatagenResult<RecordBatch> {
        let (prices, returns, sigmas) = self.simulate();
        let n = prices.len();
        let ts = timestamp_grid_ms(self.cfg.start_ms, self.cfg.step_ms, n);
        let sym = const_string(&self.cfg.symbol, n);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(ts),
            Arc::new(sym),
            Arc::new(f64_array(prices)),
            Arc::new(f64_array(returns)),
            Arc::new(f64_array(sigmas)),
        ];
        record_batch(garch_schema(), columns)
    }
}

/* -------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_seed() {
        let cfg = GarchConfig {
            seed: Some(13),
            n_steps: 100,
            ..GarchConfig::default()
        };
        let (p1, _, _) = GarchGenerator::new(cfg.clone()).unwrap().simulate();
        let (p2, _, _) = GarchGenerator::new(cfg).unwrap().simulate();
        assert_eq!(p1, p2);
    }

    #[test]
    fn sigmas_positive() {
        let g = GarchGenerator::new(GarchConfig {
            seed: Some(2),
            n_steps: 200,
            ..GarchConfig::default()
        })
        .unwrap();
        let (_, _, s) = g.simulate();
        assert!(s.iter().all(|x| *x >= 0.0 && x.is_finite()));
    }
}

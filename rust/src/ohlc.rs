//! OHLCV construction utilities.
//!
//! Given a close-price series, synthesize plausible open / high / low /
//! volume columns. The "open" is the previous close (with the first bar
//! using the first close). The high/low are drawn as symmetric envelopes
//! around the bar range scaled by an intra-bar volatility factor. Volume
//! is generated as a base level multiplied by `1 + factor * |return|`.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::arrow_util::{const_string, f64_array, record_batch, timestamp_grid_ms};
use crate::error::{DatagenError, DatagenResult};
use crate::rng::make_rng;
use crate::schema::ohlcv_schema;

#[derive(Clone, Debug)]
pub struct OhlcConfig {
    /// Intra-bar volatility (per-bar standard deviation in log space) used
    /// to draw the high/low envelope.
    pub intrabar_vol: f64,
    /// Base volume per bar.
    pub base_volume: f64,
    /// Multiplier on `|return|` that adds to volume.
    pub vol_factor: f64,
    pub symbol: String,
    pub start_ms: i64,
    pub step_ms: i64,
    pub seed: Option<u64>,
}

impl Default for OhlcConfig {
    fn default() -> Self {
        Self {
            intrabar_vol: 0.005,
            base_volume: 1_000_000.0,
            vol_factor: 5e7,
            symbol: "SYM".into(),
            start_ms: 0,
            step_ms: 86_400_000,
            seed: None,
        }
    }
}

/// Build an OHLCV `RecordBatch` from a close-price series.
pub fn ohlc_from_close(close: &[f64], cfg: &OhlcConfig) -> DatagenResult<RecordBatch> {
    if close.is_empty() {
        return Err(DatagenError::InvalidParameter(
            "close series is empty".into(),
        ));
    }
    if cfg.intrabar_vol < 0.0 || cfg.base_volume < 0.0 || cfg.vol_factor < 0.0 {
        return Err(DatagenError::InvalidParameter(
            "intrabar_vol, base_volume, vol_factor must be >= 0".into(),
        ));
    }
    let mut rng = make_rng(cfg.seed);
    let n = close.len();
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);

    for i in 0..n {
        let c = close[i];
        let o = if i == 0 { c } else { close[i - 1] };
        // Two half-normal envelopes around max(o,c) and min(o,c).
        let u_high: f64 = rng.sample::<f64, _>(StandardNormal).abs();
        let u_low: f64 = rng.sample::<f64, _>(StandardNormal).abs();
        let bar_max = o.max(c);
        let bar_min = o.min(c);
        let h = bar_max * (1.0 + cfg.intrabar_vol * u_high);
        let l = bar_min * (1.0 - cfg.intrabar_vol * u_low).max(0.0);
        let ret = if i == 0 { 0.0 } else { (c / close[i - 1]).ln() };
        let v = cfg.base_volume + cfg.vol_factor * ret.abs();

        open.push(o);
        high.push(h.max(c).max(o));
        low.push(l.min(c).min(o));
        volume.push(v);
    }

    let ts = timestamp_grid_ms(cfg.start_ms, cfg.step_ms, n);
    let sym = const_string(&cfg.symbol, n);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(ts),
        Arc::new(sym),
        Arc::new(f64_array(open)),
        Arc::new(f64_array(high)),
        Arc::new(f64_array(low)),
        Arc::new(f64_array(close.to_vec())),
        Arc::new(f64_array(volume)),
    ];
    record_batch(ohlcv_schema(), columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_invariants() {
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 0.5).collect();
        let cfg = OhlcConfig {
            seed: Some(99),
            ..OhlcConfig::default()
        };
        let rb = ohlc_from_close(&close, &cfg).unwrap();
        assert_eq!(rb.num_rows(), 20);
        assert_eq!(rb.num_columns(), 7);

        // Re-extract close, high, low and check ordering invariant.
        use arrow_array::Float64Array;
        let high = rb
            .column_by_name("high")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let low = rb
            .column_by_name("low")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for i in 0..20 {
            assert!(high.value(i) >= low.value(i));
            assert!(high.value(i) >= close[i]);
            assert!(low.value(i) <= close[i]);
        }
    }

    #[test]
    fn empty_close_errors() {
        assert!(ohlc_from_close(&[], &OhlcConfig::default()).is_err());
    }
}

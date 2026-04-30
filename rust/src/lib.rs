//! Pure-Rust core for finance-datagen.
//!
//! This crate is deliberately polars-free: every generator emits Apache
//! Arrow `RecordBatch` values. The Python bindings hand those `RecordBatch`
//! objects to Python via the Arrow C Data Interface (PyCapsule) and Python
//! constructs the polars `DataFrame` on its side. This avoids the
//! polars-Rust / polars-Python ABI mismatch.

pub mod arrow_util;
pub mod error;
pub mod ohlc;
pub mod rng;
pub mod schema;

pub mod garch;
pub mod gbm;
pub mod heston;

pub use error::{DatagenError, DatagenResult};
pub use garch::{GarchConfig, GarchGenerator};
pub use gbm::{GbmConfig, GbmGenerator};
pub use heston::{HestonConfig, HestonGenerator};
pub use ohlc::{ohlc_from_close, OhlcConfig};

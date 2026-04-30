//! Shared Arrow schemas.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, TimeUnit};

/// Schema for a single-symbol price series:
/// `[timestamp: Timestamp(ms, UTC), symbol: Utf8, price: Float64]`.
pub fn price_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

/// Schema for an OHLCV bar:
/// `[timestamp, symbol, open, high, low, close, volume]`.
pub fn ohlcv_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]))
}

/// Schema for a GARCH path: `[timestamp, symbol, price, return, sigma]`.
pub fn garch_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("return", DataType::Float64, false),
        Field::new("sigma", DataType::Float64, false),
    ]))
}

/// Schema for a Heston path: `[timestamp, symbol, price, variance]`.
pub fn heston_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("variance", DataType::Float64, false),
    ]))
}

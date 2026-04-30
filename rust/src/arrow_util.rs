//! Helpers for assembling Arrow `RecordBatch` values.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, TimestampMillisecondArray};
use arrow_schema::Schema;

use crate::error::DatagenResult;

/// Create a `Timestamp(ms, UTC)` array of `n` evenly-spaced points starting
/// at `start_ms` with `step_ms` between consecutive points.
pub fn timestamp_grid_ms(start_ms: i64, step_ms: i64, n: usize) -> TimestampMillisecondArray {
    let values: Vec<i64> = (0..n as i64).map(|i| start_ms + i * step_ms).collect();
    TimestampMillisecondArray::from(values).with_timezone("UTC")
}

/// Create a constant `Utf8` array of length `n`.
pub fn const_string(value: &str, n: usize) -> StringArray {
    StringArray::from(vec![value; n])
}

/// Convenience to build a `Float64Array` from `Vec<f64>`.
pub fn f64_array(values: Vec<f64>) -> Float64Array {
    Float64Array::from(values)
}

/// Build a `RecordBatch` from a schema and column arrays, returning a
/// `DatagenError` on schema mismatch.
pub fn record_batch(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> DatagenResult<RecordBatch> {
    Ok(RecordBatch::try_new(schema, columns)?)
}

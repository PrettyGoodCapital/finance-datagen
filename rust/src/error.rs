use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatagenError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
}

pub type DatagenResult<T> = Result<T, DatagenError>;

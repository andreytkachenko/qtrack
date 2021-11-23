use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("OnnxModel Error: {0}")]
    OnnxModelError(#[from] onnx_model::error::Error),
}

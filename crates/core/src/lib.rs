//! rsinfer-core: ORT-powered inference engine.
//!
//! Loads ONNX models (from disk or Hugging Face Hub), registers execution
//! providers, and runs text pipelines: embeddings, reranking, zero-shot
//! classification, and PII (token classification) detection.

pub mod batcher;
pub mod config;
pub mod ep;
pub mod error;
pub mod graph_pooling;
pub mod hub;
pub mod memory;
pub mod model;
pub mod pipeline;
pub mod pool;
pub mod quantize;
pub mod registry;
pub mod tokenize;

#[cfg(test)]
#[path = "tests/fixtures.rs"]
pub(crate) mod test_fixtures;

pub use config::{Config, Kind, ModelConfig};
pub use error::{Error, Result};
pub use model::LoadedModel;
pub use registry::Registry;

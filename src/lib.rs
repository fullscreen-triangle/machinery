//! # Machinery - Continuous Individual Health Modeling Through Iterative System Prediction
//!
//! Machinery is a Rust-based framework for contextual health interpretation and iterative system prediction,
//! inspired by Traditional Chinese Medicine philosophy and systems biology principles.
//!
//! ## Core Concepts
//!
//! - **Temporal Data Validity**: All health data has time-bound validity and degrades with contextual distance
//! - **Dynamic Medium Awareness**: Biological systems change while being measured
//! - **Context-Aware Predictions**: Predictions are only made when sufficient contextual similarity exists
//! - **Uncertainty Propagation**: All predictions include explicit uncertainty bounds
//!
//! ## Example Usage
//!
//! ```rust
//! use machinery::modeling::{TemporalData, MeasurementContext, DecayFunction};
//! use chrono::Duration;
//!
//! // Create temporal health data with decay function
//! let context = MeasurementContext::default();
//! let blood_pressure = TemporalData::new(
//!     120.0, // systolic BP
//!     context,
//!     DecayFunction::Exponential,
//!     Duration::hours(12), // half-life
//!     0.95, // initial confidence
//! );
//!
//! // Check current validity
//! let confidence = blood_pressure.current_confidence(None);
//! println!("Current confidence: {:.2}", confidence);
//! ```

pub use machinery_modeling as modeling;

pub mod examples;
pub mod utils;

use anyhow::Result;

/// Main Machinery health modeling system
pub struct MachinerySystem {
    /// Temporal validator for data quality
    pub validator: modeling::validation::TemporalValidator,
    /// Prediction engine
    pub predictor: modeling::prediction::AdaptiveTemporalPredictor<f64>,
}

impl MachinerySystem {
    /// Create new Machinery system with default settings
    pub fn new() -> Self {
        Self {
            validator: modeling::validation::TemporalValidator::default(),
            predictor: modeling::prediction::AdaptiveTemporalPredictor::new(
                1000, // max history size
                0.7,  // min context similarity
                0.5,  // min prediction confidence
            ),
        }
    }

    /// Process new health measurement
    pub fn process_measurement(
        &mut self,
        value: f64,
        context: modeling::MeasurementContext,
        decay_function: modeling::DecayFunction,
        confidence_half_life: chrono::Duration,
        initial_confidence: f64,
    ) -> Result<modeling::TemporalData<f64>> {
        // Create validated temporal data
        let temporal_data = self.validator.create_validated_data(
            value,
            context,
            decay_function,
            confidence_half_life,
            initial_confidence,
        )?;

        // Add to prediction engine
        self.predictor.add_data(temporal_data.clone());

        Ok(temporal_data)
    }

    /// Make prediction for specific context
    pub fn predict_for_context(
        &self,
        target_context: &modeling::MeasurementContext,
        target_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<modeling::prediction::TemporalPrediction<f64>> {
        self.predictor.predict_for_context(target_context, target_time)
    }

    /// Validate existing temporal data
    pub fn validate_data(&self, data: &modeling::TemporalData<f64>) -> modeling::validation::ValidationResult {
        self.validator.validate_data(data)
    }
}

impl Default for MachinerySystem {
    fn default() -> Self {
        Self::new()
    }
} 
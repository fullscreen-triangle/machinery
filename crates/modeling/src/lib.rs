//! Machinery Modeling Crate
//!
//! This crate implements the core temporal modeling and dynamics framework for biological systems.
//! It provides structures and algorithms for handling the time-bound nature of medical data
//! and the dynamic context-dependent validity of health measurements.

pub mod temporal;
pub mod prediction;
pub mod validation;

pub use temporal::{
    TemporalData, MeasurementContext, EnvironmentalContext, PhysiologicalContext,
    TemporalContext, MeasurementMetadata, DecayFunction, Season, TimeOfDay,
    context_similarity
};

use anyhow::Result;

/// Core trait for temporal health models
pub trait TemporalModel<T> {
    /// Make a prediction with temporal confidence bounds
    fn predict_with_confidence(&self, current_time: chrono::DateTime<chrono::Utc>) -> Result<(T, f64)>;
    
    /// Update model with new temporal data
    fn update_with_data(&mut self, data: TemporalData<T>) -> Result<()>;
    
    /// Get prediction horizon (how far into future predictions are valid)
    fn prediction_horizon(&self) -> chrono::Duration;
    
    /// Check if sufficient context similarity exists for prediction
    fn can_predict_for_context(&self, context: &MeasurementContext) -> bool;
} 
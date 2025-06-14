use crate::temporal::{TemporalData, MeasurementContext, context_similarity};
use crate::prediction::TemporalPrediction;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Validation result for temporal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
}

/// Types of validation issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationIssue {
    /// Data is too old to be reliable
    DataTooOld { age: Duration, threshold: Duration },
    /// Context similarity is too low
    ContextMismatch { similarity: f64, threshold: f64 },
    /// Confidence has decayed below threshold
    LowConfidence { confidence: f64, threshold: f64 },
    /// Missing critical context information
    MissingContext { missing_fields: Vec<String> },
    /// Prediction horizon exceeded
    PredictionHorizonExceeded { requested: Duration, max_horizon: Duration },
    /// Insufficient historical data
    InsufficientData { available: usize, required: usize },
}

/// Temporal data validator
pub struct TemporalValidator {
    /// Minimum confidence threshold for valid data
    min_confidence_threshold: f64,
    /// Maximum age for data to be considered valid
    max_data_age: Duration,
    /// Minimum context similarity for predictions
    min_context_similarity: f64,
    /// Required context fields
    required_context_fields: Vec<String>,
}

impl TemporalValidator {
    pub fn new(
        min_confidence_threshold: f64,
        max_data_age: Duration,
        min_context_similarity: f64,
    ) -> Self {
        Self {
            min_confidence_threshold,
            max_data_age,
            min_context_similarity,
            required_context_fields: vec![
                "timestamp".to_string(),
                "environmental.temperature".to_string(),
                "temporal.time_of_day".to_string(),
            ],
        }
    }

    /// Validate temporal data for current use
    pub fn validate_data<T>(&self, data: &TemporalData<T>) -> ValidationResult {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        let current_time = Utc::now();
        let data_age = current_time.signed_duration_since(data.timestamp);
        
        // Check data age
        if data_age > self.max_data_age {
            issues.push(ValidationIssue::DataTooOld {
                age: data_age,
                threshold: self.max_data_age,
            });
            recommendations.push("Consider collecting fresh data for better accuracy".to_string());
        }
        
        // Check confidence level
        let current_confidence = data.current_confidence(Some(current_time));
        if current_confidence < self.min_confidence_threshold {
            issues.push(ValidationIssue::LowConfidence {
                confidence: current_confidence,
                threshold: self.min_confidence_threshold,
            });
            recommendations.push("Data confidence has decayed - fresh measurement recommended".to_string());
        }
        
        // Check context completeness
        let missing_context = self.check_context_completeness(&data.context);
        if !missing_context.is_empty() {
            issues.push(ValidationIssue::MissingContext {
                missing_fields: missing_context.clone(),
            });
            recommendations.push(format!("Collect missing context: {}", missing_context.join(", ")));
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence: current_confidence,
            issues,
            recommendations,
        }
    }

    /// Validate prediction request
    pub fn validate_prediction_request<T>(
        &self,
        historical_data: &[TemporalData<T>],
        target_context: &MeasurementContext,
        prediction_horizon: Duration,
        max_prediction_horizon: Duration,
    ) -> ValidationResult {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // Check if sufficient historical data exists
        let min_required_data = 3;
        if historical_data.len() < min_required_data {
            issues.push(ValidationIssue::InsufficientData {
                available: historical_data.len(),
                required: min_required_data,
            });
            recommendations.push("Collect more historical data for reliable predictions".to_string());
        }
        
        // Check prediction horizon
        if prediction_horizon > max_prediction_horizon {
            issues.push(ValidationIssue::PredictionHorizonExceeded {
                requested: prediction_horizon,
                max_horizon: max_prediction_horizon,
            });
            recommendations.push("Reduce prediction horizon or collect more recent data".to_string());
        }
        
        // Check context similarity with historical data
        let mut max_similarity = 0.0;
        for data in historical_data {
            let similarity = context_similarity(&data.context, target_context);
            max_similarity = max_similarity.max(similarity);
        }
        
        if max_similarity < self.min_context_similarity {
            issues.push(ValidationIssue::ContextMismatch {
                similarity: max_similarity,
                threshold: self.min_context_similarity,
            });
            recommendations.push("Current context differs significantly from historical data".to_string());
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence: max_similarity.max(0.1), // Minimum confidence for prediction feasibility
            issues,
            recommendations,
        }
    }

    /// Validate prediction result
    pub fn validate_prediction<T>(&self, prediction: &TemporalPrediction<T>) -> ValidationResult {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // Check prediction confidence
        if prediction.confidence < self.min_confidence_threshold {
            issues.push(ValidationIssue::LowConfidence {
                confidence: prediction.confidence,
                threshold: self.min_confidence_threshold,
            });
            recommendations.push("Prediction confidence is low - use with caution".to_string());
        }
        
        // Check prediction age (how old is the prediction itself)
        let prediction_age = Utc::now().signed_duration_since(prediction.prediction_time);
        if prediction_age > Duration::hours(1) {
            recommendations.push("Prediction is over 1 hour old - consider refreshing".to_string());
        }
        
        // Check if prediction is for past time
        if prediction.target_time < Utc::now() {
            issues.push(ValidationIssue::DataTooOld {
                age: Utc::now().signed_duration_since(prediction.target_time),
                threshold: Duration::zero(),
            });
            recommendations.push("Prediction is for past time - may not be relevant".to_string());
        }
        
        ValidationResult {
            is_valid: issues.is_empty(),
            confidence: prediction.confidence,
            issues,
            recommendations,
        }
    }

    /// Check if measurement context has required fields
    fn check_context_completeness(&self, context: &MeasurementContext) -> Vec<String> {
        let mut missing = Vec::new();
        
        // Check for temperature
        if context.environmental.temperature.is_none() {
            missing.push("environmental.temperature".to_string());
        }
        
        // Check for device ID in critical measurements
        if context.measurement.device_id.is_none() {
            missing.push("measurement.device_id".to_string());
        }
        
        missing
    }

    /// Create temporal data with validation
    pub fn create_validated_data<T>(
        &self,
        value: T,
        context: MeasurementContext,
        decay_function: crate::temporal::DecayFunction,
        confidence_half_life: Duration,
        initial_confidence: f64,
    ) -> Result<TemporalData<T>> {
        // Validate context before creating data
        let missing_context = self.check_context_completeness(&context);
        if !missing_context.is_empty() {
            return Err(anyhow!("Missing required context fields: {}", missing_context.join(", ")));
        }
        
        // Validate confidence value
        if initial_confidence < 0.0 || initial_confidence > 1.0 {
            return Err(anyhow!("Initial confidence must be between 0.0 and 1.0"));
        }
        
        // Validate half-life
        if confidence_half_life <= Duration::zero() {
            return Err(anyhow!("Confidence half-life must be positive"));
        }
        
        Ok(TemporalData::new(
            value,
            context,
            decay_function,
            confidence_half_life,
            initial_confidence,
        ))
    }
}

impl Default for TemporalValidator {
    fn default() -> Self {
        Self::new(
            0.5,  // 50% minimum confidence
            Duration::days(7),  // Maximum 7 days old
            0.7,  // 70% minimum context similarity
        )
    }
}

/// Batch validation for multiple data points
pub fn batch_validate_data<T>(
    validator: &TemporalValidator,
    data_points: &[TemporalData<T>],
) -> Vec<ValidationResult> {
    data_points
        .iter()
        .map(|data| validator.validate_data(data))
        .collect()
}

/// Find the most reliable data point from a collection
pub fn find_most_reliable_data<T>(
    validator: &TemporalValidator,
    data_points: &[TemporalData<T>],
) -> Option<&TemporalData<T>> {
    data_points
        .iter()
        .map(|data| (data, validator.validate_data(data)))
        .filter(|(_, validation)| validation.is_valid)
        .max_by(|(_, a), (_, b)| a.confidence.partial_cmp(&b.confidence).unwrap())
        .map(|(data, _)| data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::{DecayFunction, MeasurementContext};

    #[test]
    fn test_validator_creation() {
        let validator = TemporalValidator::new(
            0.8,
            Duration::hours(24),
            0.9,
        );
        
        assert_eq!(validator.min_confidence_threshold, 0.8);
        assert_eq!(validator.max_data_age, Duration::hours(24));
        assert_eq!(validator.min_context_similarity, 0.9);
    }

    #[test]
    fn test_data_validation() {
        let validator = TemporalValidator::default();
        let context = MeasurementContext::default();
        
        let data = TemporalData::new(
            37.5,
            context,
            DecayFunction::Exponential,
            Duration::hours(12),
            0.9,
        );

        let result = validator.validate_data(&data);
        
        // Fresh data should be valid (though may have missing context warnings)
        assert!(result.confidence > 0.0);
    }
} 
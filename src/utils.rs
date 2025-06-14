//! Utility functions for temporal health data processing

use crate::modeling::{TemporalData, MeasurementContext, DecayFunction};
use chrono::{DateTime, Duration, Utc};
use anyhow::Result;

/// Calculate the effective "age" of data considering context changes
pub fn calculate_effective_age<T>(
    data: &TemporalData<T>,
    current_context: &MeasurementContext,
    current_time: Option<DateTime<Utc>>,
) -> Duration {
    let now = current_time.unwrap_or_else(Utc::now);
    let chronological_age = now.signed_duration_since(data.timestamp);
    
    // Context similarity affects effective age
    let context_similarity = crate::modeling::temporal::context_similarity(&data.context, current_context);
    
    // If contexts are very different, data "ages" faster
    let context_multiplier = if context_similarity < 0.5 {
        2.0 // Data ages twice as fast in different context
    } else if context_similarity < 0.8 {
        1.5 // Data ages 50% faster in somewhat different context
    } else {
        1.0 // Normal aging in similar context
    };
    
    Duration::milliseconds((chronological_age.num_milliseconds() as f64 * context_multiplier) as i64)
}

/// Create a default health measurement context for current time
pub fn create_default_health_context() -> MeasurementContext {
    let now = Utc::now();
    
    crate::modeling::MeasurementContext {
        environmental: crate::modeling::EnvironmentalContext {
            temperature: Some(22.0), // Room temperature
            humidity: Some(0.5),
            season: Some(crate::modeling::Season::from_month(now.month())),
            ..Default::default()
        },
        physiological: crate::modeling::PhysiologicalContext {
            stress_level: Some(0.3), // Moderate stress
            ..Default::default()
        },
        temporal: crate::modeling::TemporalContext {
            time_of_day: crate::modeling::TimeOfDay::from_hour(now.hour()),
            season: crate::modeling::Season::from_month(now.month()),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Determine appropriate decay function based on measurement type
pub fn suggest_decay_function(measurement_type: &str) -> (DecayFunction, Duration) {
    match measurement_type.to_lowercase().as_str() {
        "blood_pressure" => (DecayFunction::Exponential, Duration::hours(6)),
        "heart_rate" => (DecayFunction::Exponential, Duration::hours(2)),
        "temperature" => (DecayFunction::Exponential, Duration::hours(4)),
        "glucose" => (DecayFunction::Exponential, Duration::hours(8)),
        "weight" => (DecayFunction::Linear, Duration::days(7)),
        "sleep_quality" => (DecayFunction::Gaussian { sigma: Duration::hours(12) }, Duration::hours(24)),
        "stress_level" => (DecayFunction::Exponential, Duration::hours(3)),
        "exercise_performance" => (DecayFunction::Exponential, Duration::hours(12)),
        _ => (DecayFunction::Exponential, Duration::hours(6)), // Default
    }
}

/// Calculate confidence threshold based on use case
pub fn suggest_confidence_threshold(use_case: &str) -> f64 {
    match use_case.to_lowercase().as_str() {
        "emergency" => 0.3,        // Lower threshold for emergency decisions
        "clinical_decision" => 0.7, // High threshold for clinical decisions
        "wellness_tracking" => 0.5, // Moderate threshold for wellness
        "research" => 0.8,         // Very high threshold for research
        "predictive_alerts" => 0.6, // Good threshold for alerts
        _ => 0.5,                  // Default moderate threshold
    }
}

/// Format temporal data for human-readable display
pub fn format_temporal_data<T>(data: &TemporalData<T>) -> String 
where
    T: std::fmt::Display,
{
    let age = Utc::now().signed_duration_since(data.timestamp);
    let confidence = data.current_confidence(None);
    
    format!(
        "Value: {} | Age: {} | Confidence: {:.1}% | Context: {}",
        data.value,
        format_duration(age),
        confidence * 100.0,
        format_context_summary(&data.context)
    )
}

/// Format duration in human-readable form
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.num_seconds();
    
    if total_seconds < 60 {
        format!("{}s", total_seconds)
    } else if total_seconds < 3600 {
        format!("{}m", total_seconds / 60)
    } else if total_seconds < 86400 {
        format!("{}h", total_seconds / 3600)
    } else {
        format!("{}d", total_seconds / 86400)
    }
}

/// Create a summary of measurement context
pub fn format_context_summary(context: &MeasurementContext) -> String {
    let mut parts = Vec::new();
    
    // Time of day
    parts.push(format!("{:?}", context.temporal.time_of_day));
    
    // Temperature if available
    if let Some(temp) = context.environmental.temperature {
        parts.push(format!("{:.1}°C", temp));
    }
    
    // Stress level if available
    if let Some(stress) = context.physiological.stress_level {
        parts.push(format!("stress:{:.1}", stress));
    }
    
    // Time since meal if available
    if let Some(meal_time) = context.physiological.time_since_last_meal {
        parts.push(format!("meal:{}h", meal_time.num_hours()));
    }
    
    parts.join(", ")
}

/// Check if two temporal data points can be meaningfully compared
pub fn can_compare_temporal_data<T>(
    data1: &TemporalData<T>,
    data2: &TemporalData<T>,
    min_similarity: f64,
) -> bool {
    let similarity = crate::modeling::temporal::context_similarity(&data1.context, &data2.context);
    similarity >= min_similarity
}

/// Find the most recent valid data point from a collection
pub fn find_most_recent_valid<T>(
    data_points: &[TemporalData<T>],
    confidence_threshold: f64,
    current_time: Option<DateTime<Utc>>,
) -> Option<&TemporalData<T>> {
    data_points
        .iter()
        .filter(|data| data.current_confidence(current_time) >= confidence_threshold)
        .max_by_key(|data| data.timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_formatting() {
        assert_eq!(format_duration(Duration::seconds(30)), "30s");
        assert_eq!(format_duration(Duration::minutes(5)), "5m");
        assert_eq!(format_duration(Duration::hours(2)), "2h");
        assert_eq!(format_duration(Duration::days(1)), "1d");
    }

    #[test]
    fn test_decay_function_suggestions() {
        let (decay, half_life) = suggest_decay_function("blood_pressure");
        assert!(matches!(decay, DecayFunction::Exponential));
        assert_eq!(half_life, Duration::hours(6));
    }

    #[test]
    fn test_confidence_thresholds() {
        assert_eq!(suggest_confidence_threshold("emergency"), 0.3);
        assert_eq!(suggest_confidence_threshold("clinical_decision"), 0.7);
        assert_eq!(suggest_confidence_threshold("research"), 0.8);
    }
} 
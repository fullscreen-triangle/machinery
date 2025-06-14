use crate::temporal::{TemporalData, MeasurementContext, context_similarity};
use crate::TemporalModel;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Temporal prediction with confidence bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPrediction<T> {
    /// Predicted value
    pub value: T,
    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Time prediction was made
    pub prediction_time: DateTime<Utc>,
    /// Time prediction is valid for
    pub target_time: DateTime<Utc>,
    /// Context for which prediction is made
    pub target_context: MeasurementContext,
    /// Uncertainty bounds around prediction
    pub uncertainty: UncertaintyBounds<T>,
}

/// Uncertainty bounds for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds<T> {
    /// Lower bound of prediction
    pub lower_bound: Option<T>,
    /// Upper bound of prediction
    pub upper_bound: Option<T>,
    /// Standard deviation of uncertainty
    pub std_deviation: Option<f64>,
}

/// Adaptive temporal predictor that learns individual patterns
#[derive(Debug)]
pub struct AdaptiveTemporalPredictor<T> {
    /// Historical data with temporal context
    data_history: VecDeque<TemporalData<T>>,
    /// Maximum history size
    max_history_size: usize,
    /// Minimum context similarity threshold for predictions
    min_context_similarity: f64,
    /// Minimum confidence threshold for valid predictions
    min_prediction_confidence: f64,
    /// Individual-specific learning parameters
    learning_params: LearningParameters,
}

#[derive(Debug, Clone)]
pub struct LearningParameters {
    /// How quickly to adapt to new patterns
    pub adaptation_rate: f64,
    /// Weight given to recent vs historical data
    pub recency_bias: f64,
    /// Threshold for detecting pattern changes
    pub change_detection_threshold: f64,
    /// Individual-specific decay rate multiplier
    pub personal_decay_multiplier: f64,
}

impl<T> AdaptiveTemporalPredictor<T> 
where 
    T: Clone + std::fmt::Debug,
{
    pub fn new(
        max_history_size: usize,
        min_context_similarity: f64,
        min_prediction_confidence: f64,
    ) -> Self {
        Self {
            data_history: VecDeque::new(),
            max_history_size,
            min_context_similarity,
            min_prediction_confidence,
            learning_params: LearningParameters::default(),
        }
    }

    /// Add new data and learn from it
    pub fn add_data(&mut self, data: TemporalData<T>) {
        // Add to history
        self.data_history.push_back(data);
        
        // Maintain history size
        if self.data_history.len() > self.max_history_size {
            self.data_history.pop_front();
        }
        
        // Update learning parameters based on new data
        self.update_learning_parameters();
    }

    /// Make prediction for specific context and time
    pub fn predict_for_context(
        &self,
        target_context: &MeasurementContext,
        target_time: DateTime<Utc>,
    ) -> Result<TemporalPrediction<T>> {
        if self.data_history.is_empty() {
            return Err(anyhow!("No historical data available for prediction"));
        }

        // Find contextually similar historical data
        let similar_data = self.find_similar_contextual_data(target_context)?;
        
        if similar_data.is_empty() {
            return Err(anyhow!("No sufficiently similar contextual data found"));
        }

        // Calculate prediction based on similar historical patterns
        self.calculate_temporal_prediction(similar_data, target_context, target_time)
    }

    /// Find data with similar contexts
    fn find_similar_contextual_data(
        &self,
        target_context: &MeasurementContext,
    ) -> Result<Vec<&TemporalData<T>>> {
        let mut similar_data = Vec::new();
        
        for data in &self.data_history {
            let similarity = context_similarity(&data.context, target_context);
            if similarity >= self.min_context_similarity {
                similar_data.push(data);
            }
        }
        
        Ok(similar_data)
    }

    /// Calculate prediction from similar historical data
    fn calculate_temporal_prediction(
        &self,
        similar_data: Vec<&TemporalData<T>>,
        target_context: &MeasurementContext,
        target_time: DateTime<Utc>,
    ) -> Result<TemporalPrediction<T>> {
        // For now, implement a simple approach - in real implementation,
        // this would involve sophisticated temporal modeling
        
        // Find the most recent similar data point
        let most_recent = similar_data
            .iter()
            .max_by_key(|data| data.timestamp)
            .ok_or_else(|| anyhow!("No similar data found"))?;

        // Calculate confidence based on temporal distance and context similarity
        let time_distance = target_time.signed_duration_since(most_recent.timestamp);
        let context_sim = context_similarity(&most_recent.context, target_context);
        
        // Confidence decreases with time distance and increases with context similarity
        let temporal_decay = self.calculate_temporal_decay(time_distance);
        let confidence = (most_recent.current_confidence(Some(target_time)) * 
                         context_sim * temporal_decay).min(1.0);

        if confidence < self.min_prediction_confidence {
            return Err(anyhow!("Prediction confidence {} below threshold {}", 
                              confidence, self.min_prediction_confidence));
        }

        Ok(TemporalPrediction {
            value: most_recent.value.clone(),
            confidence,
            prediction_time: Utc::now(),
            target_time,
            target_context: target_context.clone(),
            uncertainty: UncertaintyBounds {
                lower_bound: None, // Would be calculated based on historical variance
                upper_bound: None,
                std_deviation: Some(1.0 - confidence), // Simple approximation
            },
        })
    }

    /// Calculate temporal decay factor
    fn calculate_temporal_decay(&self, time_distance: Duration) -> f64 {
        let hours = time_distance.num_hours() as f64;
        let decay_rate = self.learning_params.personal_decay_multiplier;
        
        // Exponential decay with personal multiplier
        (-decay_rate * hours / 24.0).exp()
    }

    /// Update learning parameters based on new data patterns
    fn update_learning_parameters(&mut self) {
        // Simple adaptive learning - would be more sophisticated in real implementation
        if self.data_history.len() >= 10 {
            // Adjust decay multiplier based on data variability
            let recent_data: Vec<_> = self.data_history.iter().rev().take(5).collect();
            
            // If recent data shows high temporal variability, increase decay rate
            let mut total_time_gaps = Duration::zero();
            for window in recent_data.windows(2) {
                let gap = window[0].timestamp.signed_duration_since(window[1].timestamp);
                total_time_gaps = total_time_gaps + gap;
            }
            
            let avg_gap_hours = total_time_gaps.num_hours() as f64 / 4.0;
            
            // Adapt decay multiplier based on data frequency
            if avg_gap_hours > 24.0 {
                self.learning_params.personal_decay_multiplier *= 1.1; // Faster decay for sparse data
            } else if avg_gap_hours < 6.0 {
                self.learning_params.personal_decay_multiplier *= 0.9; // Slower decay for frequent data
            }
            
            // Clamp to reasonable bounds
            self.learning_params.personal_decay_multiplier = 
                self.learning_params.personal_decay_multiplier.clamp(0.1, 3.0);
        }
    }

    /// Get prediction horizon based on current data patterns
    pub fn get_prediction_horizon(&self) -> Duration {
        if self.data_history.is_empty() {
            return Duration::hours(1);
        }

        // Base horizon on data frequency and confidence decay
        let avg_confidence_half_life = self.data_history
            .iter()
            .map(|d| d.confidence_half_life.num_hours())
            .sum::<i64>() as f64 / self.data_history.len() as f64;

        Duration::hours((avg_confidence_half_life * 0.5) as i64)
    }
}

impl<T> TemporalModel<T> for AdaptiveTemporalPredictor<T> 
where 
    T: Clone + std::fmt::Debug,
{
    fn predict_with_confidence(&self, current_time: DateTime<Utc>) -> Result<(T, f64)> {
        // Use default context for now - in real implementation would require current context
        let default_context = MeasurementContext::default();
        
        let prediction = self.predict_for_context(&default_context, current_time)?;
        Ok((prediction.value, prediction.confidence))
    }

    fn update_with_data(&mut self, data: TemporalData<T>) -> Result<()> {
        self.add_data(data);
        Ok(())
    }

    fn prediction_horizon(&self) -> Duration {
        self.get_prediction_horizon()
    }

    fn can_predict_for_context(&self, context: &MeasurementContext) -> bool {
        if let Ok(similar_data) = self.find_similar_contextual_data(context) {
            !similar_data.is_empty()
        } else {
            false
        }
    }
}

impl Default for LearningParameters {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            recency_bias: 0.7,
            change_detection_threshold: 0.3,
            personal_decay_multiplier: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::{DecayFunction, MeasurementContext};

    #[test]
    fn test_adaptive_predictor_creation() {
        let predictor: AdaptiveTemporalPredictor<f64> = 
            AdaptiveTemporalPredictor::new(100, 0.7, 0.5);
        
        assert_eq!(predictor.max_history_size, 100);
        assert_eq!(predictor.min_context_similarity, 0.7);
        assert_eq!(predictor.min_prediction_confidence, 0.5);
    }

    #[test]
    fn test_data_addition() {
        let mut predictor: AdaptiveTemporalPredictor<f64> = 
            AdaptiveTemporalPredictor::new(5, 0.7, 0.5);

        let context = MeasurementContext::default();
        let data = TemporalData::new(
            37.5,
            context,
            DecayFunction::Exponential,
            Duration::hours(12),
            0.9,
        );

        predictor.add_data(data);
        assert_eq!(predictor.data_history.len(), 1);
    }
} 
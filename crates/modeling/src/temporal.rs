use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Temporal data wrapper that tracks validity and decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalData<T> {
    /// The actual data value
    pub value: T,
    /// When this data was measured/collected
    pub timestamp: DateTime<Utc>,
    /// Unique identifier for this measurement
    pub id: Uuid,
    /// Context in which this measurement was taken
    pub context: MeasurementContext,
    /// How this data's validity decays over time
    pub decay_function: DecayFunction,
    /// Duration after which confidence drops to 50%
    pub confidence_half_life: Duration,
    /// Initial confidence in this measurement (0.0 to 1.0)
    pub initial_confidence: f64,
}

/// Context metadata for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementContext {
    /// Environmental factors at time of measurement
    pub environmental: EnvironmentalContext,
    /// Physiological state during measurement
    pub physiological: PhysiologicalContext,
    /// Temporal context (circadian, seasonal, etc.)
    pub temporal: TemporalContext,
    /// Measurement-specific metadata
    pub measurement: MeasurementMetadata,
    /// Custom context fields
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    pub temperature: Option<f64>,
    pub humidity: Option<f64>,
    pub air_quality_index: Option<f64>,
    pub altitude: Option<f64>,
    pub season: Option<Season>,
    pub location: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologicalContext {
    pub sleep_hours_last_night: Option<f64>,
    pub time_since_last_meal: Option<Duration>,
    pub recent_exercise: Option<ExerciseInfo>,
    pub stress_level: Option<f64>, // 0.0 to 1.0
    pub menstrual_cycle_day: Option<u8>,
    pub medications: Vec<String>,
    pub hydration_level: Option<f64>, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_of_day: TimeOfDay,
    pub day_of_week: chrono::Weekday,
    pub circadian_phase: Option<f64>, // 0.0 to 1.0 (through 24h cycle)
    pub season: Season,
    pub is_holiday: bool,
    pub timezone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetadata {
    pub device_id: Option<String>,
    pub device_accuracy: Option<f64>,
    pub measurement_duration: Option<Duration>,
    pub operator: Option<String>,
    pub calibration_date: Option<DateTime<Utc>>,
    pub measurement_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseInfo {
    pub activity: String,
    pub duration: Duration,
    pub intensity: f64, // 0.0 to 1.0
    pub time_since: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeOfDay {
    EarlyMorning,  // 4-8 AM
    Morning,       // 8-12 PM
    Afternoon,     // 12-6 PM
    Evening,       // 6-10 PM
    Night,         // 10 PM-4 AM
}

/// Different decay function types for temporal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Simple exponential decay: confidence = initial * e^(-t/half_life)
    Exponential,
    /// Linear decay: confidence = initial * (1 - t/total_lifetime)
    Linear,
    /// Step function: full confidence until threshold, then zero
    Step { threshold: Duration },
    /// Gaussian decay around measurement time
    Gaussian { sigma: Duration },
    /// Custom decay based on biological rhythms
    Circadian { 
        baseline_decay: f64,
        rhythm_amplitude: f64,
        phase_offset: Duration,
    },
}

impl<T> TemporalData<T> {
    /// Create new temporal data with current timestamp
    pub fn new(
        value: T,
        context: MeasurementContext,
        decay_function: DecayFunction,
        confidence_half_life: Duration,
        initial_confidence: f64,
    ) -> Self {
        Self {
            value,
            timestamp: Utc::now(),
            id: Uuid::new_v4(),
            context,
            decay_function,
            confidence_half_life,
            initial_confidence: initial_confidence.clamp(0.0, 1.0),
        }
    }

    /// Calculate current confidence based on elapsed time and decay function
    pub fn current_confidence(&self, current_time: Option<DateTime<Utc>>) -> f64 {
        let now = current_time.unwrap_or_else(Utc::now);
        let elapsed = now.signed_duration_since(self.timestamp);
        
        if elapsed < Duration::zero() {
            return 0.0;
        }

        self.calculate_confidence_at_time(elapsed)
    }

    /// Calculate confidence at specific time offset
    fn calculate_confidence_at_time(&self, elapsed: Duration) -> f64 {
        let elapsed_seconds = elapsed.num_seconds() as f64;
        let half_life_seconds = self.confidence_half_life.num_seconds() as f64;
        
        if half_life_seconds <= 0.0 {
            return 0.0;
        }

        let confidence = match &self.decay_function {
            DecayFunction::Exponential => {
                self.initial_confidence * (-elapsed_seconds / half_life_seconds * 0.693147).exp()
            }
            DecayFunction::Linear => {
                let total_lifetime = half_life_seconds * 2.0;
                let decay_factor = 1.0 - (elapsed_seconds / total_lifetime);
                self.initial_confidence * decay_factor.max(0.0)
            }
            DecayFunction::Step { threshold } => {
                if elapsed <= *threshold {
                    self.initial_confidence
                } else {
                    0.0
                }
            }
            DecayFunction::Gaussian { sigma } => {
                let sigma_seconds = sigma.num_seconds() as f64;
                let exponent = -(elapsed_seconds.powi(2)) / (2.0 * sigma_seconds.powi(2));
                self.initial_confidence * exponent.exp()
            }
            DecayFunction::Circadian { baseline_decay, rhythm_amplitude, phase_offset } => {
                let base_confidence = self.initial_confidence * 
                    (-elapsed_seconds / half_life_seconds * baseline_decay).exp();
                
                let phase_seconds = phase_offset.num_seconds() as f64;
                let circadian_factor = 1.0 + rhythm_amplitude * 
                    ((elapsed_seconds + phase_seconds) * 2.0 * std::f64::consts::PI / 86400.0).cos();
                
                base_confidence * circadian_factor.max(0.0)
            }
        };

        confidence.clamp(0.0, 1.0)
    }

    /// Check if data is still considered valid (confidence above threshold)
    pub fn is_valid(&self, threshold: f64, current_time: Option<DateTime<Utc>>) -> bool {
        self.current_confidence(current_time) >= threshold
    }

    /// Get age of this data
    pub fn age(&self, current_time: Option<DateTime<Utc>>) -> Duration {
        let now = current_time.unwrap_or_else(Utc::now);
        now.signed_duration_since(self.timestamp)
    }
}

/// Calculate context similarity between two measurement contexts
pub fn context_similarity(ctx1: &MeasurementContext, ctx2: &MeasurementContext) -> f64 {
    let mut similarity_scores = Vec::new();
    
    similarity_scores.push(environmental_similarity(&ctx1.environmental, &ctx2.environmental));
    similarity_scores.push(physiological_similarity(&ctx1.physiological, &ctx2.physiological));
    similarity_scores.push(temporal_similarity(&ctx1.temporal, &ctx2.temporal));
    similarity_scores.push(measurement_similarity(&ctx1.measurement, &ctx2.measurement));
    
    similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64
}

fn environmental_similarity(env1: &EnvironmentalContext, env2: &EnvironmentalContext) -> f64 {
    let mut scores = Vec::new();
    
    if let (Some(t1), Some(t2)) = (env1.temperature, env2.temperature) {
        let temp_diff = (t1 - t2).abs();
        scores.push((1.0 - (temp_diff / 10.0)).max(0.0));
    }
    
    match (&env1.season, &env2.season) {
        (Some(s1), Some(s2)) => scores.push(if s1 == s2 { 1.0 } else { 0.5 }),
        _ => {}
    }
    
    if scores.is_empty() { 0.5 } else { scores.iter().sum::<f64>() / scores.len() as f64 }
}

fn physiological_similarity(phys1: &PhysiologicalContext, phys2: &PhysiologicalContext) -> f64 {
    let mut scores = Vec::new();
    
    if let (Some(s1), Some(s2)) = (phys1.sleep_hours_last_night, phys2.sleep_hours_last_night) {
        let sleep_diff = (s1 - s2).abs();
        scores.push((1.0 - (sleep_diff / 4.0)).max(0.0));
    }
    
    if let (Some(stress1), Some(stress2)) = (phys1.stress_level, phys2.stress_level) {
        let stress_diff = (stress1 - stress2).abs();
        scores.push(1.0 - stress_diff);
    }
    
    if scores.is_empty() { 0.5 } else { scores.iter().sum::<f64>() / scores.len() as f64 }
}

fn temporal_similarity(temp1: &TemporalContext, temp2: &TemporalContext) -> f64 {
    let mut scores = Vec::new();
    
    scores.push(if temp1.time_of_day == temp2.time_of_day { 1.0 } else { 0.3 });
    scores.push(if temp1.day_of_week == temp2.day_of_week { 1.0 } else { 0.7 });
    scores.push(if temp1.season == temp2.season { 1.0 } else { 0.5 });
    
    scores.iter().sum::<f64>() / scores.len() as f64
}

fn measurement_similarity(meas1: &MeasurementMetadata, meas2: &MeasurementMetadata) -> f64 {
    let mut scores = Vec::new();
    
    match (&meas1.device_id, &meas2.device_id) {
        (Some(d1), Some(d2)) => scores.push(if d1 == d2 { 1.0 } else { 0.7 }),
        _ => {}
    }
    
    if let (Some(a1), Some(a2)) = (meas1.device_accuracy, meas2.device_accuracy) {
        let acc_diff = (a1 - a2).abs();
        scores.push((1.0 - acc_diff).max(0.0));
    }
    
    if scores.is_empty() { 0.8 } else { scores.iter().sum::<f64>() / scores.len() as f64 }
}

impl Default for MeasurementContext {
    fn default() -> Self {
        Self {
            environmental: EnvironmentalContext::default(),
            physiological: PhysiologicalContext::default(),
            temporal: TemporalContext::default(),
            measurement: MeasurementMetadata::default(),
            custom: HashMap::new(),
        }
    }
}

impl Default for EnvironmentalContext {
    fn default() -> Self {
        Self {
            temperature: None,
            humidity: None,
            air_quality_index: None,
            altitude: None,
            season: None,
            location: None,
        }
    }
}

impl Default for PhysiologicalContext {
    fn default() -> Self {
        Self {
            sleep_hours_last_night: None,
            time_since_last_meal: None,
            recent_exercise: None,
            stress_level: None,
            menstrual_cycle_day: None,
            medications: Vec::new(),
            hydration_level: None,
        }
    }
}

impl Default for TemporalContext {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            time_of_day: TimeOfDay::from_hour(now.hour()),
            day_of_week: now.weekday(),
            circadian_phase: None,
            season: Season::from_month(now.month()),
            is_holiday: false,
            timezone: "UTC".to_string(),
        }
    }
}

impl Default for MeasurementMetadata {
    fn default() -> Self {
        Self {
            device_id: None,
            device_accuracy: None,
            measurement_duration: None,
            operator: None,
            calibration_date: None,
            measurement_conditions: Vec::new(),
        }
    }
}

impl TimeOfDay {
    fn from_hour(hour: u32) -> Self {
        match hour {
            4..=7 => TimeOfDay::EarlyMorning,
            8..=11 => TimeOfDay::Morning,
            12..=17 => TimeOfDay::Afternoon,
            18..=21 => TimeOfDay::Evening,
            _ => TimeOfDay::Night,
        }
    }
}

impl Season {
    fn from_month(month: u32) -> Self {
        match month {
            3..=5 => Season::Spring,
            6..=8 => Season::Summer,
            9..=11 => Season::Autumn,
            _ => Season::Winter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let context = MeasurementContext::default();
        let data = TemporalData::new(
            42.0,
            context,
            DecayFunction::Exponential,
            Duration::hours(24),
            1.0,
        );

        assert!((data.current_confidence(Some(data.timestamp)) - 1.0).abs() < 0.01);

        let future_time = data.timestamp + Duration::hours(24);
        let confidence = data.current_confidence(Some(future_time));
        assert!((confidence - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_context_similarity() {
        let ctx1 = MeasurementContext::default();
        let ctx2 = MeasurementContext::default();
        
        let similarity = context_similarity(&ctx1, &ctx2);
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }
} 
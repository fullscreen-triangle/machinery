use crate::modeling::{
    TemporalData, MeasurementContext, EnvironmentalContext, PhysiologicalContext,
    TemporalContext, MeasurementMetadata, DecayFunction, Season, TimeOfDay,
    prediction::AdaptiveTemporalPredictor, validation::TemporalValidator,
};
use anyhow::Result;
use chrono::{Duration, Utc, DateTime};

/// Demonstrates the temporal validity problem with medical data
pub fn demonstrate_temporal_validity() -> Result<()> {
    println!("=== Temporal Validity Demonstration ===");
    
    // Create blood pressure measurement from this morning
    let morning_context = MeasurementContext {
        environmental: EnvironmentalContext {
            temperature: Some(22.0),
            humidity: Some(0.6),
            season: Some(Season::Summer),
            ..Default::default()
        },
        physiological: PhysiologicalContext {
            sleep_hours_last_night: Some(7.5),
            time_since_last_meal: Some(Duration::hours(1)),
            stress_level: Some(0.3),
            ..Default::default()
        },
        temporal: TemporalContext {
            time_of_day: TimeOfDay::Morning,
            season: Season::Summer,
            ..Default::default()
        },
        measurement: MeasurementMetadata {
            device_id: Some("BP_Monitor_001".to_string()),
            device_accuracy: Some(0.95),
            ..Default::default()
        },
        ..Default::default()
    };

    let blood_pressure = TemporalData::new(
        140.0, // Elevated systolic BP
        morning_context,
        DecayFunction::Exponential,
        Duration::hours(6), // 6-hour half-life for BP data
        0.95,
    );

    println!("Blood pressure measurement: {:.1} mmHg", blood_pressure.value);
    println!("Measured at: {}", blood_pressure.timestamp.format("%H:%M"));
    println!("Initial confidence: {:.1}%", blood_pressure.initial_confidence * 100.0);

    // Show confidence decay over time
    for hours in [0, 2, 6, 12, 24] {
        let future_time = blood_pressure.timestamp + Duration::hours(hours);
        let confidence = blood_pressure.current_confidence(Some(future_time));
        println!("After {} hours: {:.1}% confidence", hours, confidence * 100.0);
    }

    Ok(())
}

/// Demonstrates the dynamic medium effect - system changing while being measured
pub fn demonstrate_dynamic_medium_effect() -> Result<()> {
    println!("\n=== Dynamic Medium Effect Demonstration ===");
    
    let mut predictor: AdaptiveTemporalPredictor<f64> = AdaptiveTemporalPredictor::new(100, 0.7, 0.5);
    
    // Simulate heart rate measurements during exercise
    let start_time = Utc::now() - Duration::hours(2);
    
    println!("Heart rate during exercise (system changing while measuring):");
    
    for minute in 0..30 {
        let measurement_time = start_time + Duration::minutes(minute);
        
        // Heart rate increases due to exercise (dynamic system)
        let base_hr = 70.0;
        let exercise_effect = (minute as f64 * 2.0).min(40.0); // Increases up to 40 bpm
        let measured_hr = base_hr + exercise_effect;
        
        // Measurement lag - we're measuring a system that's changing
        let measurement_lag = Duration::seconds(30);
        let actual_time = measurement_time - measurement_lag;
        
        let context = MeasurementContext {
            physiological: PhysiologicalContext {
                recent_exercise: Some(crate::modeling::temporal::ExerciseInfo {
                    activity: "Running".to_string(),
                    duration: Duration::minutes(minute),
                    intensity: 0.7,
                    time_since: Duration::zero(),
                }),
                stress_level: Some(0.4 + (minute as f64 * 0.01)), // Stress increases
                ..Default::default()
            },
            ..Default::default()
        };

        let mut hr_data = TemporalData::new(
            measured_hr,
            context,
            DecayFunction::Exponential,
            Duration::minutes(5), // Short half-life during exercise
            0.90,
        );
        
        // Adjust timestamp to show measurement lag
        hr_data.timestamp = actual_time;
        
        predictor.add_data(hr_data);
        
        if minute % 5 == 0 {
            println!("  Minute {}: {:.0} bpm (system changing: +{:.0} from baseline)", 
                     minute, measured_hr, exercise_effect);
        }
    }
    
    // Try to predict current heart rate - but the "medium" (exercise state) has changed
    let current_context = MeasurementContext {
        physiological: PhysiologicalContext {
            recent_exercise: None, // Exercise stopped
            stress_level: Some(0.2), // Lower stress
            ..Default::default()
        },
        ..Default::default()
    };
    
    match predictor.predict_for_context(&current_context, Utc::now()) {
        Ok(prediction) => {
            println!("\nPrediction for current (post-exercise) state:");
            println!("  Predicted HR: {:.0} bpm", prediction.value);
            println!("  Confidence: {:.1}% (low due to context change)", prediction.confidence * 100.0);
        }
        Err(e) => {
            println!("\nPrediction failed: {}", e);
            println!("This demonstrates context dependency - can't predict without similar historical context");
        }
    }

    Ok(())
}

/// Demonstrates context-aware prediction vs naive prediction
pub fn demonstrate_context_awareness() -> Result<()> {
    println!("\n=== Context-Aware vs Naive Prediction ===");
    
    let mut predictor: AdaptiveTemporalPredictor<f64> = AdaptiveTemporalPredictor::new(100, 0.8, 0.6);
    let validator = TemporalValidator::default();
    
    // Create historical glucose measurements in different contexts
    let contexts_and_values = vec![
        // Morning fasting glucose
        (create_context(TimeOfDay::EarlyMorning, Some(Duration::hours(12)), Some(0.2)), 85.0),
        (create_context(TimeOfDay::EarlyMorning, Some(Duration::hours(11)), Some(0.2)), 87.0),
        (create_context(TimeOfDay::EarlyMorning, Some(Duration::hours(12)), Some(0.1)), 83.0),
        
        // Post-meal glucose  
        (create_context(TimeOfDay::Afternoon, Some(Duration::minutes(30)), Some(0.4)), 145.0),
        (create_context(TimeOfDay::Afternoon, Some(Duration::minutes(45)), Some(0.3)), 140.0),
        (create_context(TimeOfDay::Evening, Some(Duration::minutes(30)), Some(0.5)), 150.0),
    ];
    
    println!("Adding historical glucose measurements:");
    for (context, glucose) in &contexts_and_values {
        let data = TemporalData::new(
            *glucose,
            context.clone(),
            DecayFunction::Exponential,
            Duration::hours(8),
            0.90,
        );
        
        println!("  {:.0} mg/dL - {}, {} hours since meal", 
                 glucose, 
                 format!("{:?}", context.temporal.time_of_day),
                 context.physiological.time_since_last_meal.unwrap_or(Duration::zero()).num_hours()
        );
        
        predictor.add_data(data);
    }
    
    // Try to predict glucose for morning fasting context
    let fasting_context = create_context(TimeOfDay::EarlyMorning, Some(Duration::hours(12)), Some(0.2));
    println!("\n--- Predicting for morning fasting context ---");
    
    match predictor.predict_for_context(&fasting_context, Utc::now() + Duration::hours(1)) {
        Ok(prediction) => {
            println!("Context-aware prediction: {:.0} mg/dL (confidence: {:.1}%)", 
                     prediction.value, prediction.confidence * 100.0);
            println!("This is accurate because we have similar historical contexts");
        }
        Err(e) => println!("Prediction failed: {}", e);
    }
    
    // Try to predict for post-meal context
    let post_meal_context = create_context(TimeOfDay::Afternoon, Some(Duration::minutes(30)), Some(0.4));
    println!("\n--- Predicting for post-meal context ---");
    
    match predictor.predict_for_context(&post_meal_context, Utc::now() + Duration::hours(1)) {
        Ok(prediction) => {
            println!("Context-aware prediction: {:.0} mg/dL (confidence: {:.1}%)", 
                     prediction.value, prediction.confidence * 100.0);
            println!("Higher glucose predicted due to post-meal context");
        }
        Err(e) => println!("Prediction failed: {}", e);
    }
    
    // Try to predict for unknown context (evening exercise)
    let exercise_context = create_context(TimeOfDay::Evening, Some(Duration::hours(3)), Some(0.8));
    println!("\n--- Predicting for exercise context (no historical data) ---");
    
    match predictor.predict_for_context(&exercise_context, Utc::now() + Duration::hours(1)) {
        Ok(prediction) => {
            println!("Prediction: {:.0} mg/dL (confidence: {:.1}%)", 
                     prediction.value, prediction.confidence * 100.0);
        }
        Err(e) => {
            println!("Prediction failed: {}", e);
            println!("This demonstrates the 'cherry-picking' problem - system refuses to predict");
            println!("without sufficient contextual similarity, avoiding false confidence");
        }
    }

    Ok(())
}

/// Demonstrates the measurement lag and biological latency problem
pub fn demonstrate_measurement_latency() -> Result<()> {
    println!("\n=== Measurement Latency Demonstration ===");
    
    // Simulate the cascade: Biological Event → Physiological Change → Detectable Signal → Measurement
    let biological_event_time = Utc::now() - Duration::hours(3);
    println!("Biological event (stress response): {}", biological_event_time.format("%H:%M"));
    
    // Cortisol takes time to rise and be measurable
    let physiological_delay = Duration::minutes(30);
    let detection_delay = Duration::minutes(15);  
    let measurement_delay = Duration::minutes(10);
    
    let total_delay = physiological_delay + detection_delay + measurement_delay;
    let measurement_time = biological_event_time + total_delay;
    
    println!("Measurable cortisol level: {} (delay: {} minutes)", 
             measurement_time.format("%H:%M"), 
             total_delay.num_minutes());
    
    // But the biological system has already started adapting
    let current_time = Utc::now();
    let system_change_time = current_time.signed_duration_since(biological_event_time);
    
    println!("System has been changing for: {} hours", system_change_time.num_hours());
    println!("By the time we measure, the system is already adapting/recovering");
    
    // Create measurement with appropriate uncertainty due to latency
    let context = MeasurementContext {
        physiological: PhysiologicalContext {
            stress_level: Some(0.7), // High stress when measured
            ..Default::default()
        },
        measurement: MeasurementMetadata {
            measurement_duration: Some(Duration::minutes(5)),
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut cortisol_measurement = TemporalData::new(
        25.0, // Elevated cortisol μg/dL
        context,
        DecayFunction::Exponential,
        Duration::hours(4), // Cortisol half-life
        0.85, // Lower confidence due to measurement lag
    );
    
    // Adjust timestamp to reflect when biological event actually occurred
    cortisol_measurement.timestamp = biological_event_time;
    
    println!("\nCortisol measurement: {:.1} μg/dL", cortisol_measurement.value);
    println!("Confidence adjusted for latency: {:.1}%", cortisol_measurement.initial_confidence * 100.0);
    
    // Show how confidence accounts for the dynamic nature
    let current_confidence = cortisol_measurement.current_confidence(Some(current_time));
    println!("Current confidence (system has changed): {:.1}%", current_confidence * 100.0);

    Ok(())
}

/// Helper function to create measurement context
fn create_context(
    time_of_day: TimeOfDay, 
    time_since_meal: Option<Duration>, 
    stress_level: Option<f64>
) -> MeasurementContext {
    MeasurementContext {
        temporal: TemporalContext {
            time_of_day,
            ..Default::default()
        },
        physiological: PhysiologicalContext {
            time_since_last_meal: time_since_meal,
            stress_level,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Run all temporal dynamics demonstrations
pub fn run_all_demonstrations() -> Result<()> {
    demonstrate_temporal_validity()?;
    demonstrate_dynamic_medium_effect()?;
    demonstrate_context_awareness()?;
    demonstrate_measurement_latency()?;
    
    println!("\n=== Key Takeaways ===");
    println!("1. Medical data has time-bound validity that decays");
    println!("2. Biological systems change while being measured ('dynamic medium')");
    println!("3. Context similarity is crucial for valid predictions");
    println!("4. Measurement latency means we're always looking at the past");
    println!("5. Confidence bounds help avoid false certainty");
    
    Ok(())
} 
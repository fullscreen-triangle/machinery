#!/usr/bin/env python3
"""
Basic example demonstrating Zengeza noise reduction and attention optimization.

This example shows how to use Zengeza to reduce noise and optimize attention 
space for different types of data, particularly focusing on the classic example 
of 24-hour heart rate monitoring data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time

# Import Zengeza components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from zengeza import ZengezaEngine, AttentionMode, ZengezaConfig


def generate_noisy_heart_rate_data(duration_hours: int = 24, sampling_rate: int = 1) -> np.ndarray:
    """
    Generate synthetic 24-hour heart rate data with realistic noise.
    
    Args:
        duration_hours: Duration of monitoring in hours
        sampling_rate: Samples per second
        
    Returns:
        Noisy heart rate data array
    """
    print(f"Generating {duration_hours}-hour heart rate data...")
    
    # Time points (seconds)
    total_samples = duration_hours * 3600 * sampling_rate
    time_points = np.linspace(0, duration_hours * 3600, total_samples)
    
    # Base heart rate with circadian rhythm
    base_hr = 70  # Resting heart rate
    circadian_variation = 10 * np.sin(2 * np.pi * time_points / (24 * 3600))  # Daily cycle
    activity_spikes = 20 * np.random.exponential(0.1, total_samples) * (np.random.random(total_samples) < 0.05)  # Random activity
    
    # Clean signal
    clean_signal = base_hr + circadian_variation + activity_spikes
    
    # Add various types of noise
    gaussian_noise = np.random.normal(0, 2, total_samples)  # Sensor noise
    movement_artifacts = 5 * np.random.random(total_samples) * (np.random.random(total_samples) < 0.1)  # Movement
    baseline_drift = 3 * np.sin(2 * np.pi * time_points / 7200)  # 2-hour drift cycle
    
    # Combined noisy signal
    noisy_signal = clean_signal + gaussian_noise + movement_artifacts + baseline_drift
    
    # Ensure realistic heart rate range
    noisy_signal = np.clip(noisy_signal, 40, 200)
    
    print(f"Generated {len(noisy_signal)} data points")
    return noisy_signal


def demonstrate_attention_modes(data: np.ndarray) -> None:
    """Demonstrate different attention modes on the same data."""
    print("\n" + "="*60)
    print("DEMONSTRATING ATTENTION MODES")
    print("="*60)
    
    engine = ZengezaEngine()
    engine.start()
    
    modes = [
        (AttentionMode.CONSERVATIVE, "Conservative (preserve 80%)"),
        (AttentionMode.BALANCED, "Balanced (preserve 50%)"),
        (AttentionMode.AGGRESSIVE, "Aggressive (preserve 20%)")
    ]
    
    results = []
    
    for mode, description in modes:
        print(f"\nTesting {description}...")
        
        start_time = time.time()
        result = engine.process_data(
            data=data,
            data_type="timeseries",
            attention_mode=mode
        )
        processing_time = time.time() - start_time
        
        # Display results
        original_size = len(result.original_data)
        processed_size = len(result.processed_data) if hasattr(result.processed_data, '__len__') else 1
        compression_ratio = result.compression_ratio
        snr_db = result.noise_profile.snr_db if hasattr(result, 'noise_profile') else result.snr_db
        
        print(f"  Original size: {original_size:,} points")
        print(f"  Processed size: {processed_size:,} points")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  SNR: {snr_db:.1f} dB")
        print(f"  Processing time: {processing_time*1000:.1f} ms")
        
        results.append((mode.value, result))
    
    engine.stop()
    return results


def demonstrate_integration_context() -> None:
    """Demonstrate integration with other Machinery components."""
    print("\n" + "="*60)
    print("DEMONSTRATING MACHINERY INTEGRATION")
    print("="*60)
    
    # Create sample health data with some extraordinary events
    health_data = {
        'timestamps': list(range(0, 3600, 60)),  # 1 hour, 1-minute intervals
        'heart_rate': [72 + np.random.normal(0, 2) for _ in range(60)],
        'blood_pressure_systolic': [120 + np.random.normal(0, 5) for _ in range(60)],
    }
    
    # Add some extraordinary events (cardiac abnormalities)
    health_data['heart_rate'][30] = 180  # Tachycardia event
    health_data['heart_rate'][31] = 185
    health_data['blood_pressure_systolic'][30] = 170
    health_data['blood_pressure_systolic'][31] = 175
    
    # Create context with extraordinary events
    context = {
        'extraordinary_events': [
            {
                'timestamp': 30,
                'extraordinarity_level': 0.9,
                'type': 'cardiac_event',
                'description': 'Sudden tachycardia with hypertension'
            }
        ],
        'patient_risk_level': 'high',
        'monitoring_priority': 'critical'
    }
    
    # Configure engine with integration settings
    config = ZengezaConfig()
    config.integration_settings['mzekezeke']['enabled'] = True
    config.integration_settings['spectacular']['enabled'] = True
    
    engine = ZengezaEngine(config)
    engine.start()
    
    print("Processing health data with extraordinary events...")
    
    # Process heart rate data
    hr_result = engine.process_data(
        data=health_data['heart_rate'],
        data_type='health',
        attention_mode=AttentionMode.BALANCED
    )
    
    print(f"Heart Rate Processing:")
    print(f"  Original: {len(health_data['heart_rate'])} points")
    print(f"  Processed: {len(hr_result.processed_data)} points")
    print(f"  Compression: {hr_result.compression_ratio:.2f}x")
    print(f"  SNR: {hr_result.snr_db:.1f} dB")
    
    # In a real implementation, the integration would:
    # 1. Detect the extraordinary event (Spectacular)
    # 2. Preserve more data around the cardiac event
    # 3. Apply health-specific compression (Mzekezeke)
    # 4. Optimize for downstream health analysis
    
    engine.stop()


def demonstrate_real_world_scenario() -> None:
    """Demonstrate a real-world scenario: IoT sensor network optimization."""
    print("\n" + "="*60)
    print("REAL-WORLD SCENARIO: IoT SENSOR NETWORK")
    print("="*60)
    
    print("Scenario: Environmental monitoring network with 100 sensors")
    print("Challenge: Reduce bandwidth usage while preserving important events")
    
    # Simulate sensor data from multiple environmental sensors
    sensors = {
        'temperature': np.random.normal(22, 3, 1440),  # 24 hours, 1-minute intervals
        'humidity': np.random.normal(60, 10, 1440),
        'air_quality': np.random.normal(50, 15, 1440),
        'noise_level': np.random.normal(45, 8, 1440),
    }
    
    # Add some environmental events
    # Pollution spike at hour 10
    sensors['air_quality'][600:630] += 40
    # Temperature spike at hour 15 (heat event)
    sensors['temperature'][900:920] += 8
    # Noise event at hour 18 (traffic/construction)
    sensors['noise_level'][1080:1140] += 20
    
    engine = ZengezaEngine()
    engine.start()
    
    total_original_size = 0
    total_processed_size = 0
    total_processing_time = 0
    
    print("\nProcessing sensor data streams...")
    
    for sensor_type, data in sensors.items():
        start_time = time.time()
        
        # Use adaptive mode to handle different sensor characteristics
        result = engine.process_data(
            data=data,
            data_type="timeseries",
            attention_mode=AttentionMode.ADAPTIVE
        )
        
        processing_time = time.time() - start_time
        
        original_size = len(data)
        processed_size = len(result.processed_data) if hasattr(result.processed_data, '__len__') else len(data)
        
        total_original_size += original_size
        total_processed_size += processed_size
        total_processing_time += processing_time
        
        print(f"  {sensor_type.capitalize()}:")
        print(f"    Original: {original_size} points")
        print(f"    Processed: {processed_size} points")
        print(f"    Compression: {result.compression_ratio:.2f}x")
        print(f"    Processing: {processing_time*1000:.1f} ms")
    
    # Calculate overall statistics
    overall_compression = total_original_size / total_processed_size if total_processed_size > 0 else 1.0
    bandwidth_savings = (1 - 1/overall_compression) * 100 if overall_compression > 1 else 0
    
    print(f"\nOverall Network Optimization:")
    print(f"  Total original data: {total_original_size:,} points")
    print(f"  Total processed data: {total_processed_size:,} points")
    print(f"  Overall compression: {overall_compression:.2f}x")
    print(f"  Bandwidth savings: {bandwidth_savings:.1f}%")
    print(f"  Total processing time: {total_processing_time*1000:.1f} ms")
    
    engine.stop()


def main():
    """Main demonstration function."""
    print("="*60)
    print("ZENGEZA NOISE REDUCTION & ATTENTION OPTIMIZATION")
    print("Making processes tenable by reducing the attention space")
    print("="*60)
    
    # Generate sample data
    print("\nGenerating sample data...")
    heart_rate_data = generate_noisy_heart_rate_data(duration_hours=1, sampling_rate=1)  # 1 hour for demo
    
    print(f"Sample statistics:")
    print(f"  Mean heart rate: {np.mean(heart_rate_data):.1f} bpm")
    print(f"  Standard deviation: {np.std(heart_rate_data):.1f} bpm")
    print(f"  Range: {np.min(heart_rate_data):.1f} - {np.max(heart_rate_data):.1f} bpm")
    
    # Demonstrate different attention modes
    results = demonstrate_attention_modes(heart_rate_data)
    
    # Demonstrate integration features
    demonstrate_integration_context()
    
    # Demonstrate real-world scenario
    demonstrate_real_world_scenario()
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Zengeza can significantly reduce data size while preserving important information")
    print("• Different attention modes provide flexibility for various use cases")
    print("• Integration with other Machinery components enhances processing intelligence")
    print("• Real-world applications show substantial bandwidth and storage savings")
    print("\nFor more examples and advanced usage, see the documentation.")
    
    return results


if __name__ == "__main__":
    # Run the demonstration
    try:
        results = main()
        print("\n✅ Demonstration completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 
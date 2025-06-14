"""
Mzekezeke: Scientific Health Analysis and ML Prediction Engine

A Python framework for applying scientific methods to health data analysis,
machine learning predictions, and biomedical signal processing. Designed to
complement the Machinery Rust framework for temporal health modeling.

Core Modules:
- health_metrics: Analysis of vital signs and health indicators
- ml_models: HuggingFace and custom ML model integration
- scientific_methods: Implementation of medical/biological analysis methods
- api: REST API for integration with external systems
"""

__version__ = "0.1.0"
__author__ = "Machinery Team"

# Core imports
from .health_metrics import (
    HeartRateAnalyzer,
    BloodPressureAnalyzer,
    RespirationAnalyzer,
    SleepAnalyzer,
    StressAnalyzer,
)

from .ml_models import (
    HuggingFacePredictor,
    HealthOutcomePredictor,
    AnomalyDetector,
    RiskAssessment,
)

from .scientific_methods import (
    StatisticalAnalysis,
    SignalProcessing,
    BiometricsCalculator,
    ClinicalScoring,
)

from .core import (
    ProcessingEngine,
    HealthDataProcessor,
    PredictionPipeline,
)

# Convenience classes
__all__ = [
    # Core engine
    "ProcessingEngine",
    "HealthDataProcessor", 
    "PredictionPipeline",
    
    # Health metrics
    "HeartRateAnalyzer",
    "BloodPressureAnalyzer",
    "RespirationAnalyzer",
    "SleepAnalyzer", 
    "StressAnalyzer",
    
    # ML models
    "HuggingFacePredictor",
    "HealthOutcomePredictor",
    "AnomalyDetector",
    "RiskAssessment",
    
    # Scientific methods
    "StatisticalAnalysis",
    "SignalProcessing",
    "BiometricsCalculator",
    "ClinicalScoring",
] 
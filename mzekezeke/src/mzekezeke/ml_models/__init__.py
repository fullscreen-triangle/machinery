"""
Machine Learning Models Module

This module provides ML model interfaces for health predictions including:
- HuggingFace model integration for health language models
- Custom health outcome prediction models
- Anomaly detection for health data
- Risk assessment models
- Time series forecasting for health metrics
"""

from .huggingface import HuggingFacePredictor
from .health_outcomes import HealthOutcomePredictor
from .anomaly_detection import AnomalyDetector
from .risk_assessment import RiskAssessment
from .base import MLModelInterface

__all__ = [
    "MLModelInterface",
    "HuggingFacePredictor",
    "HealthOutcomePredictor", 
    "AnomalyDetector",
    "RiskAssessment",
] 
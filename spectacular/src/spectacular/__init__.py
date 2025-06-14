"""
Spectacular: Extraordinary Information Detection and Prioritization System

This module provides sophisticated algorithms for identifying, classifying,
and prioritizing extraordinary information nodes that require special attention
in complex systems.

Key Features:
- Anomaly detection using multiple algorithms
- Information theory-based extraordinarity scoring
- Network analysis for exceptional node identification
- Adaptive thresholding systems
- Real-time streaming analysis
- Integration with Machinery framework components

The module is designed to work seamlessly with:
- mzekezeke: Health metrics analysis and ML predictions
- diggiden: Adversarial system for challenging optimization
- hatata: Markov Decision Processes and stochastic methods
"""

from .core import SpectacularEngine
from .detectors import (
    AnomalyDetector,
    OutlierDetector,
    NoveltyDetector,
    InformationTheoryDetector,
)
from .scorers import (
    ExtraordinaryScorer,
    InformationContentScorer,
    RarityScorer,
    ImpactScorer,
)
from .networks import (
    NetworkAnalyzer,
    GraphAnomalyDetector,
    CommunityDetector,
    InfluenceAnalyzer,
)
from .processors import (
    StreamProcessor,
    BatchProcessor,
    AdaptiveProcessor,
    RealTimeProcessor,
)
from .visualizers import (
    SpectacularVisualizer,
    AnomalyPlotter,
    NetworkPlotter,
    InteractiveDashboard,
)
from .integrations import (
    MzekezekeIntegration,
    DiggidenIntegration,
    HatataIntegration,
    MachineryIntegration,
)
from .config import SpectacularConfig
from .utils import (
    data_utils,
    math_utils,
    validation_utils,
    metrics_utils,
)

__version__ = "0.1.0"
__author__ = "Machinery Team"
__email__ = "team@machinery.dev"

__all__ = [
    # Core engine
    "SpectacularEngine",
    
    # Detection modules
    "AnomalyDetector",
    "OutlierDetector", 
    "NoveltyDetector",
    "InformationTheoryDetector",
    
    # Scoring systems
    "ExtraordinaryScorer",
    "InformationContentScorer",
    "RarityScorer",
    "ImpactScorer",
    
    # Network analysis
    "NetworkAnalyzer",
    "GraphAnomalyDetector",
    "CommunityDetector",
    "InfluenceAnalyzer",
    
    # Processing engines
    "StreamProcessor",
    "BatchProcessor",
    "AdaptiveProcessor", 
    "RealTimeProcessor",
    
    # Visualization
    "SpectacularVisualizer",
    "AnomalyPlotter",
    "NetworkPlotter",
    "InteractiveDashboard",
    
    # Integrations
    "MzekezekeIntegration",
    "DiggidenIntegration",
    "HatataIntegration",
    "MachineryIntegration",
    
    # Configuration
    "SpectacularConfig",
    
    # Utilities
    "data_utils",
    "math_utils",
    "validation_utils",
    "metrics_utils",
] 
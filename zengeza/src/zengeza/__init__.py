"""
Zengeza: Noise Reduction and Attention Space Optimization System

This module provides sophisticated noise reduction and attention focusing
capabilities for data processing in the Machinery framework. Zengeza identifies
and filters out information that doesn't contribute meaningfully to processing,
allowing systems to focus on the signal rather than the noise.

Key Features:
- Noise detection and quantification
- Signal-to-noise ratio optimization
- Attention space reduction
- Information density analysis
- Adaptive filtering strategies
- Integration with Machinery framework components

Philosophy:
Not all information is equally valuable. Even valid data can contain noise that
doesn't contribute to the processing goal. Zengeza makes processes more tenable
by reducing the attention space to focus on what truly matters.

Example Use Cases:
- Time series data compression (e.g., 24-hour heart rate data)
- Feature selection and dimensionality reduction
- Information filtering for large datasets
- Attention optimization for neural networks
- Signal processing in noisy environments
"""

from .core import ZengezaEngine, NoiseLevel, AttentionMode
from .detectors import (
    NoiseDetector,
    RedundancyDetector,
    SignalDetector,
    InformationDensityDetector,
    TemporalNoiseDetector,
    SpatialNoiseDetector,
)
from .filters import (
    NoiseFilter,
    AdaptiveFilter,
    StatisticalFilter,
    WaveletFilter,
    KalmanFilter,
    MedianFilter,
    GaussianFilter,
)
from .reducers import (
    AttentionReducer,
    DimensionalityReducer,
    CompressionReducer,
    FeatureSelector,
    SamplingReducer,
    ClusteringReducer,
)
from .analyzers import (
    NoiseAnalyzer,
    SignalAnalyzer,
    InformationAnalyzer,
    ComplexityAnalyzer,
    EntropyAnalyzer,
    SpectrumAnalyzer,
)
from .optimizers import (
    AttentionOptimizer,
    SNROptimizer,
    CompressionOptimizer,
    EfficiencyOptimizer,
    AdaptiveOptimizer,
)
from .processors import (
    StreamProcessor,
    BatchProcessor,
    TimeSeriesProcessor,
    ImageProcessor,
    TextProcessor,
    SignalProcessor,
)
from .integrations import (
    MzekezekeIntegration,
    SpectacularIntegration,
    NicotineIntegration,
    DiggidenIntegration,
    HatataIntegration,
    MachineryIntegration,
)
from .config import ZengezaConfig
from .utils import (
    noise_utils,
    signal_utils,
    compression_utils,
    attention_utils,
    analysis_utils,
)

__version__ = "0.1.0"
__author__ = "Machinery Team"
__email__ = "team@machinery.dev"

__all__ = [
    # Core engine
    "ZengezaEngine",
    "NoiseLevel",
    "AttentionMode",
    
    # Detection systems
    "NoiseDetector",
    "RedundancyDetector",
    "SignalDetector",
    "InformationDensityDetector",
    "TemporalNoiseDetector",
    "SpatialNoiseDetector",
    
    # Filtering systems
    "NoiseFilter",
    "AdaptiveFilter",
    "StatisticalFilter", 
    "WaveletFilter",
    "KalmanFilter",
    "MedianFilter",
    "GaussianFilter",
    
    # Attention reduction
    "AttentionReducer",
    "DimensionalityReducer",
    "CompressionReducer",
    "FeatureSelector",
    "SamplingReducer",
    "ClusteringReducer",
    
    # Analysis tools
    "NoiseAnalyzer",
    "SignalAnalyzer",
    "InformationAnalyzer",
    "ComplexityAnalyzer",
    "EntropyAnalyzer",
    "SpectrumAnalyzer",
    
    # Optimization engines
    "AttentionOptimizer",
    "SNROptimizer",
    "CompressionOptimizer",
    "EfficiencyOptimizer",
    "AdaptiveOptimizer",
    
    # Processing systems
    "StreamProcessor",
    "BatchProcessor",
    "TimeSeriesProcessor",
    "ImageProcessor",
    "TextProcessor",
    "SignalProcessor",
    
    # Integrations
    "MzekezekeIntegration",
    "SpectacularIntegration",
    "NicotineIntegration",
    "DiggidenIntegration",
    "HatataIntegration",
    "MachineryIntegration",
    
    # Configuration
    "ZengezaConfig",
    
    # Utilities
    "noise_utils",
    "signal_utils",
    "compression_utils",
    "attention_utils",
    "analysis_utils",
] 
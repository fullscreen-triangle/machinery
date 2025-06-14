"""
Configuration module for the Zengeza framework.

This module contains configuration classes and default settings
for noise reduction and attention space optimization.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class ZengezaConfig:
    """Main configuration class for the Zengeza system."""
    
    # Core engine settings
    max_history_size: int = 1000
    max_noise_profiles: int = 500
    max_attention_maps: int = 500
    processing_timeout_seconds: float = 30.0
    
    # Noise detection settings
    noise_detection_method: str = "statistical"  # statistical, spectral, adaptive
    snr_calculation_method: str = "power_ratio"  # power_ratio, peak_to_peak, rms
    noise_threshold_db: float = 10.0
    minimum_signal_level: float = 0.001
    
    # Filtering settings
    filter_types: List[str] = field(default_factory=lambda: ["gaussian", "median", "kalman"])
    adaptive_filtering: bool = True
    filter_window_size: int = 5
    filter_sigma: float = 1.0
    
    # Attention optimization settings
    default_attention_mode: str = "balanced"
    compression_targets: Dict[str, float] = field(default_factory=lambda: {
        "conservative": 0.8,  # 20% compression
        "balanced": 0.5,      # 50% compression  
        "aggressive": 0.2     # 80% compression
    })
    importance_calculation_method: str = "variance"  # variance, gradient, frequency, ml
    minimum_attention_preservation: float = 0.1  # Always preserve at least 10%
    
    # Data type specific settings
    timeseries_settings: Dict[str, Any] = field(default_factory=lambda: {
        'window_size': 10,
        'overlap_ratio': 0.5,
        'trend_removal': True,
        'seasonal_adjustment': False,
        'outlier_detection': True,
        'sampling_strategies': ['uniform', 'importance', 'adaptive']
    })
    
    image_settings: Dict[str, Any] = field(default_factory=lambda: {
        'patch_size': 32,
        'stride': 16,
        'edge_preservation': True,
        'texture_analysis': True,
        'saliency_detection': True,
        'compression_methods': ['jpeg', 'wavelet', 'roi_based']
    })
    
    text_settings: Dict[str, Any] = field(default_factory=lambda: {
        'tokenization_method': 'word',
        'stopword_removal': True,
        'stemming': False,
        'importance_metrics': ['tf_idf', 'attention_weights', 'semantic_similarity'],
        'preservation_strategies': ['key_sentences', 'summarization', 'keyword_extraction']
    })
    
    signal_settings: Dict[str, Any] = field(default_factory=lambda: {
        'fft_window': 'hann',
        'frequency_bands': [(0, 50), (50, 200), (200, 1000)],
        'spectral_analysis': True,
        'peak_detection': True,
        'harmonic_analysis': False
    })
    
    # Quality metrics settings
    quality_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'calculate_mse': True,
        'calculate_psnr': True,
        'calculate_ssim': True,
        'calculate_mutual_info': True,
        'calculate_compression_efficiency': True,
        'perceptual_quality_assessment': False
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enable_parallel_processing': True,
        'max_worker_threads': 4,
        'memory_limit_mb': 1024,
        'enable_caching': True,
        'cache_size': 100,
        'enable_gpu_acceleration': False
    })
    
    # Integration settings
    integration_settings: Dict[str, Any] = field(default_factory=lambda: {
        'mzekezeke': {
            'enabled': True,
            'health_data_compression': 0.3,
            'vital_signs_preservation': 0.8,
            'anomaly_detection_integration': True
        },
        'spectacular': {
            'enabled': True,
            'extraordinary_event_preservation': 0.9,
            'noise_threshold_adjustment': True,
            'attention_boost_for_events': 2.0
        },
        'nicotine': {
            'enabled': True,
            'context_data_optimization': True,
            'puzzle_data_compression': 0.6,
            'validation_data_preservation': 0.8
        },
        'diggiden': {
            'enabled': True,
            'adversarial_noise_detection': True,
            'defense_signal_enhancement': True,
            'attack_pattern_filtering': True
        },
        'hatata': {
            'enabled': True,
            'decision_data_compression': 0.4,
            'state_transition_preservation': 0.7,
            'policy_gradient_optimization': True
        }
    })
    
    # Adaptive learning settings
    adaptive_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enable_learning': True,
        'learning_rate': 0.01,
        'adaptation_window': 100,
        'performance_feedback': True,
        'parameter_optimization': True,
        'model_update_frequency': 50
    })
    
    # Logging and monitoring
    logging_settings: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': None,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'log_processing_stats': True,
        'log_noise_profiles': False,
        'log_attention_maps': False
    })
    
    # Validation settings
    validation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'validate_input_data': True,
        'validate_output_quality': True,
        'quality_threshold': 0.8,
        'reconstruction_error_threshold': 0.1,
        'compression_limit_check': True,
        'performance_monitoring': True
    })
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'ZengezaConfig':
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    @classmethod
    def load_from_env(cls) -> 'ZengezaConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if 'ZENGEZA_NOISE_THRESHOLD' in os.environ:
            config.noise_threshold_db = float(os.environ['ZENGEZA_NOISE_THRESHOLD'])
        
        if 'ZENGEZA_DEFAULT_ATTENTION_MODE' in os.environ:
            config.default_attention_mode = os.environ['ZENGEZA_DEFAULT_ATTENTION_MODE']
        
        if 'ZENGEZA_FILTER_WINDOW_SIZE' in os.environ:
            config.filter_window_size = int(os.environ['ZENGEZA_FILTER_WINDOW_SIZE'])
        
        if 'ZENGEZA_PROCESSING_TIMEOUT' in os.environ:
            config.processing_timeout_seconds = float(os.environ['ZENGEZA_PROCESSING_TIMEOUT'])
        
        return config
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a JSON file."""
        config_dict = self.to_dict()
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate thresholds
        if self.noise_threshold_db <= 0:
            issues.append("noise_threshold_db must be positive")
        
        if not 0.0 <= self.minimum_attention_preservation <= 1.0:
            issues.append("minimum_attention_preservation must be between 0.0 and 1.0")
        
        # Validate compression targets
        for mode, target in self.compression_targets.items():
            if not 0.0 < target <= 1.0:
                issues.append(f"compression_target for {mode} must be between 0.0 and 1.0")
        
        # Validate window sizes
        if self.filter_window_size <= 0:
            issues.append("filter_window_size must be positive")
        
        if self.processing_timeout_seconds <= 0:
            issues.append("processing_timeout_seconds must be positive")
        
        # Validate history sizes
        if self.max_history_size <= 0:
            issues.append("max_history_size must be positive")
        
        return issues
    
    def update(self, **kwargs) -> 'ZengezaConfig':
        """Update configuration with new values."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)
    
    def get_compression_target(self, attention_mode: str) -> float:
        """Get compression target for a specific attention mode."""
        return self.compression_targets.get(attention_mode, 0.5)
    
    def get_data_type_config(self, data_type: str) -> Dict[str, Any]:
        """Get configuration for specific data type."""
        config_map = {
            'timeseries': self.timeseries_settings,
            'image': self.image_settings,
            'text': self.text_settings,
            'signal': self.signal_settings
        }
        return config_map.get(data_type, {})
    
    def is_integration_enabled(self, integration_name: str) -> bool:
        """Check if a specific integration is enabled."""
        integration_config = self.integration_settings.get(integration_name, {})
        return integration_config.get('enabled', False)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.performance_settings.copy()
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality metrics configuration."""
        return self.quality_metrics.copy()


@dataclass
class NoiseDetectionConfig:
    """Configuration for noise detection algorithms."""
    method: str = "statistical"
    window_size: int = 10
    threshold_multiplier: float = 2.0
    adaptive_threshold: bool = True
    frequency_analysis: bool = True
    
    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")


@dataclass
class AttentionConfig:
    """Configuration for attention optimization."""
    mode: str = "balanced"
    importance_method: str = "variance"
    compression_ratio: float = 0.5
    preserve_edges: bool = True
    adaptive_threshold: bool = True
    
    def __post_init__(self):
        if not 0.0 < self.compression_ratio <= 1.0:
            raise ValueError("compression_ratio must be between 0.0 and 1.0")


# Default configuration instance
DEFAULT_CONFIG = ZengezaConfig() 
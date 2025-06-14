"""
Configuration module for the Spectacular framework.

This module contains configuration classes and default settings
for the extraordinary information detection system.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class SpectacularConfig:
    """Main configuration class for the Spectacular system."""
    
    # Core engine settings
    max_workers: int = 4
    enable_async_processing: bool = True
    processing_timeout: float = 30.0
    
    # Extraordinarity thresholds (0.0 to 1.0)
    extraordinarity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'extraordinary': 0.95,
        'exceptional': 0.85, 
        'rare': 0.70,
        'unusual': 0.50,
        'notable': 0.30
    })
    
    # Minimum threshold to consider something extraordinary
    min_extraordinarity_threshold: float = 0.30
    
    # Scorer weights for combining different scoring algorithms
    scorer_weights: Dict[str, float] = field(default_factory=lambda: {
        'information_content': 1.5,
        'rarity': 1.2,
        'impact': 1.3,
        'anomaly': 1.0,
        'novelty': 0.8
    })
    
    # Detector settings
    detector_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'anomaly_detector': {
            'enabled': True,
            'algorithms': ['isolation_forest', 'one_class_svm', 'local_outlier_factor'],
            'contamination': 0.1,
            'n_estimators': 100
        },
        'outlier_detector': {
            'enabled': True,
            'methods': ['z_score', 'iqr', 'modified_z_score'],
            'z_threshold': 3.0,
            'iqr_multiplier': 1.5
        },
        'novelty_detector': {
            'enabled': True,
            'window_size': 100,
            'similarity_threshold': 0.8
        },
        'information_theory_detector': {
            'enabled': True,
            'entropy_threshold': 0.7,
            'mutual_info_threshold': 0.5
        }
    })
    
    # Network analysis settings
    network_settings: Dict[str, Any] = field(default_factory=lambda: {
        'community_detection': {
            'algorithm': 'louvain',
            'resolution': 1.0
        },
        'influence_analysis': {
            'centrality_measures': ['betweenness', 'closeness', 'eigenvector'],
            'propagation_model': 'independent_cascade'
        },
        'graph_anomaly': {
            'structural_measures': ['degree', 'clustering', 'pagerank'],
            'temporal_window': 10
        }
    })
    
    # Processing settings
    processing_settings: Dict[str, Any] = field(default_factory=lambda: {
        'batch_size': 1000,
        'buffer_size': 10000,
        'flush_interval': 60.0,  # seconds
        'enable_streaming': True,
        'stream_window_size': 1000
    })
    
    # Integration settings
    integration_settings: Dict[str, Any] = field(default_factory=lambda: {
        'mzekezeke': {
            'enabled': True,
            'health_threshold_multiplier': 1.2,
            'ml_confidence_weight': 0.3
        },
        'diggiden': {
            'enabled': True,
            'adversarial_boost': 1.5,
            'challenge_response_weight': 0.4
        },
        'hatata': {
            'enabled': True,
            'mdp_state_importance': 0.6,
            'stochastic_variation_penalty': 0.2
        }
    })
    
    # Visualization settings
    visualization_settings: Dict[str, Any] = field(default_factory=lambda: {
        'default_theme': 'dark',
        'max_nodes_display': 1000,
        'interactive_updates': True,
        'export_formats': ['png', 'svg', 'html'],
        'color_schemes': {
            'extraordinarity': ['#2E8B57', '#FF6347', '#FF4500', '#DC143C', '#8B0000'],
            'confidence': ['#ADD8E6', '#4169E1', '#0000FF', '#00008B', '#191970']
        }
    })
    
    # Logging and monitoring
    logging_settings: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': None,  # Will use default if None
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        'cache_size': 1000,
        'enable_gpu': False,
        'memory_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cpu_limit': 0.8  # 80% of available cores
    })
    
    # Data persistence
    persistence_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enable_persistence': True,
        'storage_backend': 'sqlite',  # or 'postgresql', 'mongodb'
        'connection_string': None,
        'auto_save_interval': 300,  # seconds
        'retention_days': 30
    })
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'SpectacularConfig':
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    @classmethod
    def load_from_env(cls) -> 'SpectacularConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if 'SPECTACULAR_MAX_WORKERS' in os.environ:
            config.max_workers = int(os.environ['SPECTACULAR_MAX_WORKERS'])
        
        if 'SPECTACULAR_MIN_THRESHOLD' in os.environ:
            config.min_extraordinarity_threshold = float(os.environ['SPECTACULAR_MIN_THRESHOLD'])
        
        if 'SPECTACULAR_ENABLE_GPU' in os.environ:
            config.performance_settings['enable_gpu'] = os.environ['SPECTACULAR_ENABLE_GPU'].lower() == 'true'
        
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
        for name, threshold in self.extraordinarity_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                issues.append(f"Threshold '{name}' must be between 0.0 and 1.0, got {threshold}")
        
        # Validate weights
        for name, weight in self.scorer_weights.items():
            if weight < 0:
                issues.append(f"Scorer weight '{name}' must be non-negative, got {weight}")
        
        # Validate worker count
        if self.max_workers <= 0:
            issues.append(f"max_workers must be positive, got {self.max_workers}")
        
        # Validate timeout
        if self.processing_timeout <= 0:
            issues.append(f"processing_timeout must be positive, got {self.processing_timeout}")
        
        return issues
    
    def update(self, **kwargs) -> 'SpectacularConfig':
        """Update configuration with new values."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)


@dataclass
class DetectorConfig:
    """Configuration for individual detectors."""
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Detector weight must be non-negative")


@dataclass  
class ScorerConfig:
    """Configuration for individual scorers."""
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    normalization: str = 'minmax'  # 'minmax', 'zscore', 'robust'
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Scorer weight must be non-negative")
        if self.normalization not in ['minmax', 'zscore', 'robust']:
            raise ValueError(f"Invalid normalization method: {self.normalization}")


# Default configuration instance
DEFAULT_CONFIG = SpectacularConfig() 
"""
Configuration module for the Nicotine framework.

This module contains configuration classes and default settings
for the context validation and coherence maintenance system.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class NicotineConfig:
    """Main configuration class for the Nicotine system."""
    
    # Core engine settings
    max_memory_size: int = 100
    max_snapshots: int = 50
    monitoring_interval_seconds: float = 5.0
    
    # Break trigger settings
    break_trigger_process_count: int = 25
    break_trigger_time_seconds: float = 300.0  # 5 minutes
    break_trigger_coherence_threshold: float = 0.3
    
    # Puzzle settings
    min_puzzle_confidence: float = 0.7
    puzzle_timeout_seconds: float = 30.0
    puzzle_difficulty_levels: List[str] = field(default_factory=lambda: ["simple", "easy", "normal", "hard", "expert"])
    
    # Context settings
    context_window_size: int = 10
    context_hash_algorithm: str = "md5"
    context_refresh_threshold: float = 0.5
    
    # Coherence tracking
    coherence_history_size: int = 100
    coherence_degradation_window: int = 5
    coherence_critical_threshold: float = 0.2
    
    # Puzzle generation settings
    puzzle_generation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'context_puzzle': {
            'enabled': True,
            'difficulty_adaptive': True,
            'memory_depth': 5,
            'question_types': ['recall', 'inference', 'synthesis', 'validation']
        },
        'logic_puzzle': {
            'enabled': True,
            'complexity_levels': ['basic', 'intermediate', 'advanced'],
            'problem_types': ['deduction', 'induction', 'pattern_matching']
        },
        'memory_puzzle': {
            'enabled': True,
            'recall_depth': 10,
            'sequence_lengths': [3, 5, 7, 10],
            'pattern_types': ['sequential', 'categorical', 'hierarchical']
        },
        'summary_puzzle': {
            'enabled': True,
            'summarization_levels': ['sentence', 'paragraph', 'document'],
            'compression_ratios': [0.1, 0.25, 0.5]
        }
    })
    
    # Solver settings
    solver_settings: Dict[str, Any] = field(default_factory=lambda: {
        'timeout_seconds': 30.0,
        'max_attempts': 3,
        'confidence_threshold': 0.7,
        'use_parallel_solving': True,
        'solver_strategies': ['exhaustive', 'heuristic', 'ml_based']
    })
    
    # Scheduling settings
    scheduling_settings: Dict[str, Any] = field(default_factory=lambda: {
        'adaptive_scheduling': True,
        'base_interval_minutes': 5,
        'max_interval_minutes': 60,
        'stress_factor_multiplier': 0.5,
        'success_factor_multiplier': 1.2
    })
    
    # Integration settings
    integration_settings: Dict[str, Any] = field(default_factory=lambda: {
        'mzekezeke': {
            'enabled': True,
            'health_context_weight': 0.8,
            'critical_health_break_trigger': True
        },
        'spectacular': {
            'enabled': True,
            'extraordinary_event_break_trigger': True,
            'extraordinariness_threshold': 0.8
        },
        'diggiden': {
            'enabled': True,
            'adversarial_detection_weight': 0.6,
            'challenge_mode_enabled': True
        },
        'hatata': {
            'enabled': True,
            'decision_context_integration': True,
            'mdp_state_tracking': True
        }
    })
    
    # Validation settings
    validation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'context_integrity_checks': True,
        'cross_reference_validation': True,
        'temporal_consistency_checks': True,
        'semantic_coherence_analysis': True,
        'validation_confidence_threshold': 0.8
    })
    
    # Performance settings
    performance_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enable_caching': True,
        'cache_size': 1000,
        'enable_compression': True,
        'max_concurrent_puzzles': 3,
        'memory_optimization': True
    })
    
    # Logging and monitoring
    logging_settings: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': None,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'log_context_snapshots': True,
        'log_puzzle_attempts': True
    })
    
    # Security settings
    security_settings: Dict[str, Any] = field(default_factory=lambda: {
        'encrypt_context_data': False,
        'secure_puzzle_generation': True,
        'context_data_retention_hours': 24,
        'secure_memory_cleanup': True
    })
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'NicotineConfig':
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    @classmethod
    def load_from_env(cls) -> 'NicotineConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if 'NICOTINE_BREAK_TRIGGER_COUNT' in os.environ:
            config.break_trigger_process_count = int(os.environ['NICOTINE_BREAK_TRIGGER_COUNT'])
        
        if 'NICOTINE_BREAK_TRIGGER_TIME' in os.environ:
            config.break_trigger_time_seconds = float(os.environ['NICOTINE_BREAK_TRIGGER_TIME'])
        
        if 'NICOTINE_MIN_PUZZLE_CONFIDENCE' in os.environ:
            config.min_puzzle_confidence = float(os.environ['NICOTINE_MIN_PUZZLE_CONFIDENCE'])
        
        if 'NICOTINE_MONITORING_INTERVAL' in os.environ:
            config.monitoring_interval_seconds = float(os.environ['NICOTINE_MONITORING_INTERVAL'])
        
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
        
        # Validate break triggers
        if self.break_trigger_process_count <= 0:
            issues.append("break_trigger_process_count must be positive")
        
        if self.break_trigger_time_seconds <= 0:
            issues.append("break_trigger_time_seconds must be positive")
        
        # Validate confidence thresholds
        if not 0.0 <= self.min_puzzle_confidence <= 1.0:
            issues.append("min_puzzle_confidence must be between 0.0 and 1.0")
        
        if not 0.0 <= self.break_trigger_coherence_threshold <= 1.0:
            issues.append("break_trigger_coherence_threshold must be between 0.0 and 1.0")
        
        # Validate memory settings
        if self.max_memory_size <= 0:
            issues.append("max_memory_size must be positive")
        
        if self.max_snapshots <= 0:
            issues.append("max_snapshots must be positive")
        
        # Validate timeouts
        if self.puzzle_timeout_seconds <= 0:
            issues.append("puzzle_timeout_seconds must be positive")
        
        if self.monitoring_interval_seconds <= 0:
            issues.append("monitoring_interval_seconds must be positive")
        
        return issues
    
    def update(self, **kwargs) -> 'NicotineConfig':
        """Update configuration with new values."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)
    
    def get_break_trigger_config(self) -> Dict[str, Any]:
        """Get break trigger configuration."""
        return {
            'process_count': self.break_trigger_process_count,
            'time_seconds': self.break_trigger_time_seconds,
            'coherence_threshold': self.break_trigger_coherence_threshold
        }
    
    def get_puzzle_config(self, puzzle_type: str) -> Dict[str, Any]:
        """Get configuration for specific puzzle type."""
        return self.puzzle_generation_settings.get(puzzle_type, {})
    
    def is_integration_enabled(self, integration_name: str) -> bool:
        """Check if a specific integration is enabled."""
        integration_config = self.integration_settings.get(integration_name, {})
        return integration_config.get('enabled', False)


@dataclass
class PuzzleConfig:
    """Configuration for individual puzzle types."""
    enabled: bool = True
    difficulty_level: str = "normal"
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    confidence_threshold: float = 0.7
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


@dataclass
class BreakScheduleConfig:
    """Configuration for break scheduling."""
    enabled: bool = True
    adaptive: bool = True
    base_interval_minutes: float = 5.0
    max_interval_minutes: float = 60.0
    stress_multiplier: float = 0.5
    success_multiplier: float = 1.2
    
    def __post_init__(self):
        if self.base_interval_minutes <= 0:
            raise ValueError("base_interval_minutes must be positive")
        if self.max_interval_minutes <= self.base_interval_minutes:
            raise ValueError("max_interval_minutes must be greater than base_interval_minutes")


# Default configuration instance
DEFAULT_CONFIG = NicotineConfig() 
"""
Utility functions for the Spectacular framework.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DataUtils:
    """Utilities for data processing and manipulation."""
    
    @staticmethod
    def normalize_data(data: Union[List, np.ndarray], method: str = 'minmax') -> np.ndarray:
        """Normalize data using specified method."""
        data_array = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            if max_val > min_val:
                return (data_array - min_val) / (max_val - min_val)
            else:
                return data_array
        elif method == 'zscore':
            mean_val = np.mean(data_array)
            std_val = np.std(data_array)
            if std_val > 0:
                return (data_array - mean_val) / std_val
            else:
                return data_array - mean_val
        else:
            return data_array
    
    @staticmethod
    def extract_features(data: Any) -> Dict[str, Any]:
        """Extract basic features from data."""
        features = {}
        
        if isinstance(data, (list, tuple)):
            features['length'] = len(data)
            features['type'] = 'sequence'
        elif isinstance(data, dict):
            features['length'] = len(data)
            features['type'] = 'mapping'
        elif isinstance(data, str):
            features['length'] = len(data)
            features['type'] = 'string'
        else:
            features['length'] = 1
            features['type'] = type(data).__name__
        
        return features


class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """Calculate Shannon entropy."""
        import math
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float], metric: str = 'euclidean') -> float:
        """Calculate distance between two points."""
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        if metric == 'euclidean':
            return np.linalg.norm(p1 - p2)
        elif metric == 'manhattan':
            return np.sum(np.abs(p1 - p2))
        else:
            return np.linalg.norm(p1 - p2)  # Default to euclidean


class ValidationUtils:
    """Utilities for data validation."""
    
    @staticmethod
    def validate_score(score: float) -> bool:
        """Validate that score is between 0 and 1."""
        return 0.0 <= score <= 1.0
    
    @staticmethod
    def validate_data_format(data: Any, expected_type: type) -> bool:
        """Validate data format."""
        return isinstance(data, expected_type)


class MetricsUtils:
    """Utilities for calculating metrics."""
    
    @staticmethod
    def calculate_precision_recall(true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
        """Calculate precision and recall."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        }


# Expose utility modules
data_utils = DataUtils()
math_utils = MathUtils()
validation_utils = ValidationUtils()
metrics_utils = MetricsUtils() 
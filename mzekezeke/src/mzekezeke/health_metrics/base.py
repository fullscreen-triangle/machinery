"""
Base classes for health metric analysis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd


class HealthMetricAnalyzer(ABC):
    """Base class for all health metric analyzers."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.supported_metrics = set()
        
    @abstractmethod
    def analyze(self, data: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze health data and return results.
        
        Args:
            data: List of health data points
            context: Additional context for analysis
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: List[Any]) -> bool:
        """Validate that the input data is appropriate for this analyzer."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get analyzer metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "supported_metrics": list(self.supported_metrics),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def preprocess_data(self, data: List[Any]) -> np.ndarray:
        """Preprocess data for analysis."""
        # Convert to numpy array for numerical analysis
        if isinstance(data[0], dict):
            # Handle complex data structures
            values = []
            for item in data:
                if isinstance(item.value, (int, float)):
                    values.append(item.value)
                elif isinstance(item.value, dict) and 'value' in item.value:
                    values.append(item.value['value'])
            return np.array(values)
        else:
            return np.array([item.value for item in data])
    
    def calculate_basic_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        if len(values) == 0:
            return {}
            
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
            "variance": float(np.var(values)),
            "percentile_25": float(np.percentile(values, 25)),
            "percentile_75": float(np.percentile(values, 75)),
        }
    
    def detect_outliers(self, values: np.ndarray, method: str = "iqr") -> List[int]:
        """Detect outliers in the data."""
        if len(values) < 4:
            return []
            
        outlier_indices = []
        
        if method == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = [i for i, v in enumerate(values) 
                             if v < lower_bound or v > upper_bound]
        
        elif method == "zscore":
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outlier_indices = [i for i, z in enumerate(z_scores) if z > 3]
        
        return outlier_indices
    
    def calculate_trend(self, values: np.ndarray, timestamps: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """Calculate trend in the data."""
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Classify trend
        trend_threshold = np.std(values) * 0.1  # 10% of std as threshold
        
        if abs(slope) < trend_threshold:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": float(slope),
            "intercept": float(intercept),
            "trend_strength": abs(slope) / np.std(values) if np.std(values) > 0 else 0
        } 
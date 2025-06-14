"""
Base ML model interface for health predictions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np


class MLModelInterface(ABC):
    """Base interface for all ML models in mzekezeke."""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.is_trained = False
        self.last_updated = datetime.now()
        self.model_metadata = {}
        
    @abstractmethod
    def predict(
        self, 
        data: List[Any], 
        prediction_horizon: timedelta, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make predictions based on input data.
        
        Args:
            data: Input health data
            prediction_horizon: How far into the future to predict
            context: Additional context for prediction
            
        Returns:
            Prediction results
        """
        pass
    
    @abstractmethod
    def train(self, training_data: List[Any], labels: List[Any]) -> Dict[str, Any]:
        """
        Train the model with provided data.
        
        Args:
            training_data: Training input data
            labels: Training labels/targets
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: List[Any]) -> bool:
        """Validate input data format and quality."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.model_metadata
        }
    
    def preprocess_data(self, data: List[Any]) -> np.ndarray:
        """Preprocess data for model input."""
        # Basic preprocessing - extract numerical values
        if not data:
            return np.array([])
        
        processed = []
        for item in data:
            if hasattr(item, 'value'):
                if isinstance(item.value, (int, float)):
                    processed.append(item.value)
                elif isinstance(item.value, dict):
                    # Handle complex health data structures
                    processed.append(list(item.value.values()))
            else:
                processed.append(item)
        
        return np.array(processed)
    
    def calculate_prediction_confidence(
        self, 
        data: List[Any], 
        prediction: Any, 
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for prediction."""
        base_confidence = 0.8
        
        # Reduce confidence based on data quality
        if len(data) < 10:
            base_confidence *= 0.7
        
        # Reduce confidence if model is not recently updated
        days_since_update = (datetime.now() - self.last_updated).days
        if days_since_update > 30:
            base_confidence *= 0.9
        
        return max(0.1, min(1.0, base_confidence))
    
    def update_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update model metadata."""
        self.model_metadata.update(metadata)
        self.last_updated = datetime.now() 
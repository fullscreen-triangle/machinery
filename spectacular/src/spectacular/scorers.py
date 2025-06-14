"""
Scoring algorithms for quantifying extraordinariness.

This module implements various scoring mechanisms to quantify how extraordinary
a piece of information is, including information content, rarity, and impact scores.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import math
from collections import defaultdict, Counter


class BaseScorer(ABC):
    """Base class for all scorers."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.weight = self.config.get('weight', 1.0)
    
    @abstractmethod
    async def score(self, data: Dict[str, Any]) -> float:
        """
        Calculate score for given data.
        
        Args:
            data: Detection result to score
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class ExtraordinaryScorer(BaseScorer):
    """Combined scorer that aggregates multiple scoring methods."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("extraordinary_scorer", config)
        self.sub_scorers = []
        self.normalization_method = self.config.get('normalization', 'minmax')
    
    def add_scorer(self, scorer: BaseScorer) -> None:
        """Add a sub-scorer to the ensemble."""
        self.sub_scorers.append(scorer)
    
    async def score(self, data: Dict[str, Any]) -> float:
        """Calculate combined extraordinarity score."""
        if not self.sub_scorers:
            return 0.5  # Neutral score if no sub-scorers
        
        scores = []
        for scorer in self.sub_scorers:
            try:
                score = await scorer.score(data)
                scores.append(score * scorer.weight)
            except Exception as e:
                self.logger.error(f"Scorer {scorer.name} failed: {str(e)}")
                scores.append(0.0)
        
        if not scores:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(scorer.weight for scorer in self.sub_scorers)
        return sum(scores) / total_weight if total_weight > 0 else 0.0


class InformationContentScorer(BaseScorer):
    """Score based on information content and entropy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("information_content_scorer", config)
        self.entropy_weight = self.config.get('entropy_weight', 0.6)
        self.complexity_weight = self.config.get('complexity_weight', 0.4)
    
    async def score(self, data: Dict[str, Any]) -> float:
        """Calculate information content score."""
        try:
            raw_data = data.get('data', '')
            
            # Calculate entropy
            entropy_score = self._calculate_entropy(raw_data)
            
            # Calculate complexity
            complexity_score = self._calculate_complexity(raw_data)
            
            # Combine scores
            final_score = (
                entropy_score * self.entropy_weight +
                complexity_score * self.complexity_weight
            )
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Information content scoring failed: {str(e)}")
            return 0.0
    
    def _calculate_entropy(self, data: Any) -> float:
        """Calculate normalized Shannon entropy."""
        try:
            text = str(data)
            if not text:
                return 0.0
            
            # Character frequency analysis
            char_counts = Counter(text)
            total_chars = len(text)
            
            # Calculate entropy
            entropy = 0.0
            for count in char_counts.values():
                probability = count / total_chars
                entropy -= probability * math.log2(probability)
            
            # Normalize by maximum possible entropy
            max_entropy = math.log2(len(char_counts)) if len(char_counts) > 1 else 1
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Entropy calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_complexity(self, data: Any) -> float:
        """Calculate structural complexity score."""
        try:
            if isinstance(data, dict):
                # Dictionary complexity: depth + key diversity
                depth = self._get_dict_depth(data)
                key_diversity = len(set(str(k) for k in data.keys())) / len(data) if data else 0
                return min(1.0, (depth / 10) * 0.5 + key_diversity * 0.5)
            
            elif isinstance(data, (list, tuple)):
                # List complexity: length variation + type diversity
                if not data:
                    return 0.0
                type_diversity = len(set(type(x).__name__ for x in data)) / len(data)
                length_factor = min(1.0, len(data) / 100)
                return type_diversity * 0.7 + length_factor * 0.3
            
            elif isinstance(data, str):
                # String complexity: character diversity + structure
                if not data:
                    return 0.0
                char_diversity = len(set(data)) / len(data)
                structure_score = self._analyze_string_structure(data)
                return char_diversity * 0.6 + structure_score * 0.4
            
            else:
                # Generic complexity based on string representation
                text = str(data)
                return min(1.0, len(set(text)) / len(text)) if text else 0.0
                
        except Exception as e:
            self.logger.error(f"Complexity calculation failed: {str(e)}")
            return 0.0
    
    def _get_dict_depth(self, d: dict, depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(d, dict):
            return depth
        
        max_depth = depth
        for value in d.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, self._get_dict_depth(value, depth + 1))
        
        return max_depth
    
    def _analyze_string_structure(self, text: str) -> float:
        """Analyze structural complexity of string."""
        try:
            # Look for patterns, repetitions, and structure
            words = text.split()
            sentences = text.split('.')
            
            # Word length variation
            if words:
                word_lengths = [len(word) for word in words]
                word_variation = np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) > 0 else 0
            else:
                word_variation = 0
            
            # Sentence structure
            sentence_variation = len(sentences) / len(text) if text else 0
            
            return min(1.0, word_variation * 0.5 + sentence_variation * 0.5)
            
        except Exception:
            return 0.0


class RarityScorer(BaseScorer):
    """Score based on statistical rarity and frequency."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("rarity_scorer", config)
        self.frequency_memory = defaultdict(int)
        self.total_observations = 0
        self.novelty_bonus = self.config.get('novelty_bonus', 0.2)
    
    async def score(self, data: Dict[str, Any]) -> float:
        """Calculate rarity score."""
        try:
            # Create signature for the data
            signature = self._create_signature(data.get('data'))
            
            # Update frequency memory
            self.frequency_memory[signature] += 1
            self.total_observations += 1
            
            # Calculate rarity score
            frequency = self.frequency_memory[signature]
            rarity = 1.0 - (frequency / self.total_observations)
            
            # Apply novelty bonus for first-time observations
            if frequency == 1:
                rarity += self.novelty_bonus
                rarity = min(1.0, rarity)
            
            return rarity
            
        except Exception as e:
            self.logger.error(f"Rarity scoring failed: {str(e)}")
            return 0.5  # Neutral score on error
    
    def _create_signature(self, data: Any) -> str:
        """Create a signature for data to track frequency."""
        try:
            if isinstance(data, dict):
                # Create signature from keys and value types
                items = sorted([(k, type(v).__name__) for k, v in data.items()])
                return f"dict:{hash(str(items))}"
            
            elif isinstance(data, (list, tuple)):
                # Create signature from length and element types
                type_counts = Counter(type(x).__name__ for x in data)
                signature = f"list:{len(data)}:{hash(str(sorted(type_counts.items())))}"
                return signature
            
            elif isinstance(data, str):
                # Create signature from length and character patterns
                char_types = {
                    'alpha': sum(1 for c in data if c.isalpha()),
                    'digit': sum(1 for c in data if c.isdigit()),
                    'space': sum(1 for c in data if c.isspace()),
                    'punct': sum(1 for c in data if not c.isalnum() and not c.isspace())
                }
                return f"str:{len(data)}:{hash(str(sorted(char_types.items())))}"
            
            else:
                # Generic signature
                return f"{type(data).__name__}:{hash(str(data))}"
                
        except Exception as e:
            # Fallback signature
            return f"unknown:{hash(str(data))}"


class ImpactScorer(BaseScorer):
    """Score based on potential impact and influence."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("impact_scorer", config)
        self.context_weight = self.config.get('context_weight', 0.4)
        self.magnitude_weight = self.config.get('magnitude_weight', 0.6)
    
    async def score(self, data: Dict[str, Any]) -> float:
        """Calculate impact score."""
        try:
            # Analyze context importance
            context_score = self._analyze_context(data)
            
            # Analyze magnitude/scale
            magnitude_score = self._analyze_magnitude(data)
            
            # Combine scores
            final_score = (
                context_score * self.context_weight +
                magnitude_score * self.magnitude_weight
            )
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Impact scoring failed: {str(e)}")
            return 0.0
    
    def _analyze_context(self, data: Dict[str, Any]) -> float:
        """Analyze contextual importance."""
        try:
            # Look for context indicators
            context_indicators = [
                'critical', 'urgent', 'important', 'alert', 'warning',
                'error', 'failure', 'success', 'breakthrough', 'significant'
            ]
            
            text = str(data.get('data', '')).lower()
            
            # Count context indicators
            indicator_count = sum(1 for indicator in context_indicators if indicator in text)
            
            # Normalize by text length and indicator count
            context_density = indicator_count / len(context_indicators)
            
            # Consider detection method confidence
            confidence = data.get('confidence', 0.5)
            
            return min(1.0, context_density * 0.7 + confidence * 0.3)
            
        except Exception:
            return 0.0
    
    def _analyze_magnitude(self, data: Dict[str, Any]) -> float:
        """Analyze magnitude/scale of the data."""
        try:
            raw_data = data.get('data')
            
            # Numerical magnitude
            if isinstance(raw_data, (int, float)):
                # Use logarithmic scaling
                magnitude = min(1.0, abs(raw_data) / 1000000)  # Scale to millions
                return magnitude
            
            # Collection magnitude
            elif isinstance(raw_data, (list, tuple, dict)):
                size = len(raw_data)
                magnitude = min(1.0, size / 10000)  # Scale to 10k items
                return magnitude
            
            # Text magnitude
            elif isinstance(raw_data, str):
                length = len(raw_data)
                magnitude = min(1.0, length / 100000)  # Scale to 100k chars
                return magnitude
            
            # Default magnitude based on string representation
            else:
                text_length = len(str(raw_data))
                return min(1.0, text_length / 1000)
                
        except Exception:
            return 0.0


class AnomalyScorer(BaseScorer):
    """Score specifically for anomaly detection results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("anomaly_scorer", config)
    
    async def score(self, data: Dict[str, Any]) -> float:
        """Score anomaly detection results."""
        try:
            # Use existing anomaly scores if available
            if 'anomaly_score' in data:
                return min(1.0, max(0.0, float(data['anomaly_score'])))
            
            # Use detection confidence
            if 'confidence' in data:
                return min(1.0, max(0.0, float(data['confidence'])))
            
            # Default moderate score for anomalies
            return 0.7
            
        except Exception as e:
            self.logger.error(f"Anomaly scoring failed: {str(e)}")
            return 0.0 
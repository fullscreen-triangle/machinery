"""Utility functions for the Zengeza framework."""

import logging
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


class NoiseUtils:
    """Utilities for noise analysis and handling."""
    
    @staticmethod
    def estimate_noise_variance(data: np.ndarray) -> float:
        """Estimate noise variance using robust methods."""
        if len(data) < 2:
            return 0.1
        
        # Use median absolute deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (mad * 1.4826) ** 2  # Convert to variance
    
    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray = None) -> float:
        """Calculate signal-to-noise ratio."""
        signal_power = np.var(signal)
        
        if noise is not None:
            noise_power = np.var(noise)
        else:
            # Estimate noise from signal
            noise_power = NoiseUtils.estimate_noise_variance(signal)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return 10 * np.log10(max(snr_linear, 1e-10))
        else:
            return 40.0  # High SNR if no noise


class SignalUtils:
    """Utilities for signal processing."""
    
    @staticmethod
    def find_peaks(data: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in signal data."""
        if len(data) < 3:
            return []
        
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and 
                data[i] > data[i+1] and 
                data[i] > threshold):
                peaks.append(i)
        
        return peaks
    
    @staticmethod
    def smooth_signal(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply smoothing to signal."""
        if len(data) < window_size:
            return data
        
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')


class CompressionUtils:
    """Utilities for data compression and reduction."""
    
    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    @staticmethod
    def select_important_indices(
        importance_scores: np.ndarray, 
        keep_ratio: float
    ) -> np.ndarray:
        """Select most important indices based on scores."""
        n_keep = max(1, int(len(importance_scores) * keep_ratio))
        return np.argsort(importance_scores)[-n_keep:]
    
    @staticmethod
    def uniform_sampling(data: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniformly sample data to target size."""
        if len(data) <= target_size:
            return data, np.arange(len(data))
        
        indices = np.linspace(0, len(data) - 1, target_size, dtype=int)
        return data[indices], indices


class AttentionUtils:
    """Utilities for attention mechanism."""
    
    @staticmethod
    def calculate_importance_scores(data: np.ndarray, method: str = "variance") -> np.ndarray:
        """Calculate importance scores for data segments."""
        if method == "variance":
            return AttentionUtils._variance_importance(data)
        elif method == "gradient":
            return AttentionUtils._gradient_importance(data)
        else:
            return np.ones(len(data))
    
    @staticmethod
    def _variance_importance(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate importance based on local variance."""
        importance = np.zeros(len(data))
        
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            local_data = data[start:end]
            importance[i] = np.var(local_data) if len(local_data) > 1 else 0.0
        
        return importance
    
    @staticmethod
    def _gradient_importance(data: np.ndarray) -> np.ndarray:
        """Calculate importance based on gradient magnitude."""
        if len(data) < 2:
            return np.ones(len(data))
        
        gradients = np.abs(np.gradient(data))
        return gradients / (np.max(gradients) + 1e-8)
    
    @staticmethod
    def create_attention_mask(
        importance_scores: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """Create binary attention mask from importance scores."""
        normalized_scores = importance_scores / (np.max(importance_scores) + 1e-8)
        return (normalized_scores >= threshold).astype(float)


class AnalysisUtils:
    """Utilities for data analysis."""
    
    @staticmethod
    def calculate_entropy(data: np.ndarray, bins: int = 50) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(data, bins=bins, density=True)
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        # Calculate entropy
        return -np.sum(probabilities * np.log2(probabilities))
    
    @staticmethod
    def calculate_complexity(data: np.ndarray) -> float:
        """Calculate complexity measure of data."""
        if len(data) < 2:
            return 0.0
        
        # Use normalized compression ratio as complexity measure
        # Higher compression ratio = lower complexity
        try:
            # Simple complexity based on autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Find where autocorrelation drops below threshold
            threshold_idx = np.where(autocorr < 0.1)[0]
            if len(threshold_idx) > 0:
                complexity = 1.0 / (threshold_idx[0] + 1)
            else:
                complexity = 1.0 / len(autocorr)
            
            return min(complexity, 1.0)
        except:
            return 0.5  # Default complexity
    
    @staticmethod
    def calculate_information_density(data: np.ndarray) -> float:
        """Calculate information density of data."""
        if len(data) == 0:
            return 0.0
        
        # Combine entropy and complexity measures
        entropy = AnalysisUtils.calculate_entropy(data)
        complexity = AnalysisUtils.calculate_complexity(data)
        
        # Weighted combination
        density = 0.7 * entropy + 0.3 * complexity
        return min(density / 10.0, 1.0)  # Normalize to [0, 1]


# Module-level utility instances for easy access
noise_utils = NoiseUtils()
signal_utils = SignalUtils()
compression_utils = CompressionUtils()
attention_utils = AttentionUtils()
analysis_utils = AnalysisUtils() 
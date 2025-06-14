"""Utility functions for the Nicotine framework."""

import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class ContextUtils:
    """Utilities for context management."""
    
    @staticmethod
    def hash_context(context: Dict[str, Any]) -> str:
        """Generate a hash for context data."""
        try:
            context_str = json.dumps(context, sort_keys=True, default=str)
            return hashlib.md5(context_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()
    
    @staticmethod
    def merge_contexts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two context dictionaries."""
        merged = base.copy()
        merged.update(update)
        return merged
    
    @staticmethod
    def validate_context_structure(context: Dict[str, Any]) -> bool:
        """Validate basic context structure."""
        required_keys = ['timestamp', 'process_count']
        return all(key in context for key in required_keys)


class PuzzleUtils:
    """Utilities for puzzle management."""
    
    @staticmethod
    def generate_puzzle_id(puzzle_type: str) -> str:
        """Generate a unique puzzle ID."""
        timestamp = datetime.now().isoformat()
        content = f"{puzzle_type}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def calculate_puzzle_difficulty(context: Dict[str, Any]) -> str:
        """Calculate appropriate puzzle difficulty."""
        coherence_score = context.get('coherence_score', 0.5)
        
        if coherence_score >= 0.8:
            return "hard"
        elif coherence_score >= 0.6:
            return "normal"
        elif coherence_score >= 0.4:
            return "easy"
        else:
            return "simple"


class ValidationUtils:
    """Utilities for validation operations."""
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """Validate confidence score range."""
        return 0.0 <= score <= 1.0
    
    @staticmethod
    def calculate_weighted_score(scores: List[Tuple[float, float]]) -> float:
        """Calculate weighted average score."""
        if not scores:
            return 0.0
        
        total_weighted = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)
        
        return total_weighted / total_weight if total_weight > 0 else 0.0


class SchedulingUtils:
    """Utilities for scheduling operations."""
    
    @staticmethod
    def calculate_next_break_time(
        last_break: datetime, 
        base_interval: float, 
        stress_factor: float = 1.0
    ) -> datetime:
        """Calculate next break time."""
        from datetime import timedelta
        
        adjusted_interval = base_interval * stress_factor
        return last_break + timedelta(minutes=adjusted_interval)
    
    @staticmethod
    def should_adaptive_schedule(
        success_rate: float, 
        threshold: float = 0.8
    ) -> bool:
        """Determine if adaptive scheduling should be used."""
        return success_rate >= threshold


# Utility instances
context_utils = ContextUtils()
puzzle_utils = PuzzleUtils()
validation_utils = ValidationUtils()
scheduling_utils = SchedulingUtils() 
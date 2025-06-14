"""Schedulers for managing context break timing."""

import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class BaseScheduler(ABC):
    """Base class for break schedulers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def should_schedule_break(self, context: Dict[str, Any]) -> bool:
        """Determine if a break should be scheduled."""
        pass


class BreakScheduler(BaseScheduler):
    """Basic break scheduler."""
    
    def should_schedule_break(self, context: Dict[str, Any]) -> bool:
        """Basic break scheduling logic."""
        return False  # Placeholder


class AdaptiveScheduler(BaseScheduler):
    """Adaptive break scheduler that adjusts based on system state."""
    
    def should_schedule_break(self, context: Dict[str, Any]) -> bool:
        """Adaptive break scheduling logic."""
        return False  # Placeholder


class ProcessCountScheduler(BaseScheduler):
    """Schedules breaks based on process count."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.trigger_count = self.config.get('trigger_count', 25)
    
    def should_schedule_break(self, context: Dict[str, Any]) -> bool:
        """Schedule break based on process count."""
        process_count = context.get('process_count', 0)
        return process_count >= self.trigger_count


class TimeBasedScheduler(BaseScheduler):
    """Schedules breaks based on time intervals."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.interval_minutes = self.config.get('interval_minutes', 5.0)
        self.last_break = datetime.now()
    
    def should_schedule_break(self, context: Dict[str, Any]) -> bool:
        """Schedule break based on time elapsed."""
        elapsed = datetime.now() - self.last_break
        return elapsed >= timedelta(minutes=self.interval_minutes) 
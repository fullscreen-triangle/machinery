"""
Nicotine: Context Validation and Coherence Maintenance System

This module provides "cigarette break" functionality for AI systems - periodic
context validation through machine-readable puzzles to ensure the system
maintains understanding of its purpose and context.

Key Features:
- Context tracking and validation
- Machine-readable puzzle generation and solving
- System coherence verification
- Context summarization and refresh
- Adaptive break scheduling
- Integration with Machinery framework components

The module is designed to prevent context drift and maintain system coherence
in long-running AI processes by periodically challenging the system to prove
its understanding through puzzle-solving.

Philosophy:
Just as humans take cigarette breaks to step back and refocus, AI systems
need periodic "context breaks" to validate their understanding and refresh
their mental model of the current task.
"""

from .core import NicotineEngine, ContextState, CoherenceLevel
from .puzzles import (
    PuzzleGenerator,
    ContextPuzzle,
    LogicPuzzle,
    MemoryPuzzle,
    SummaryPuzzle,
    IntegrationPuzzle,
)
from .context import (
    ContextTracker,
    ContextSummarizer,
    ContextValidator,
    ContextRefresher,
)
from .schedulers import (
    BreakScheduler,
    AdaptiveScheduler,
    ProcessCountScheduler,
    TimeBasedScheduler,
)
from .solvers import (
    PuzzleSolver,
    LogicSolver,
    MemorySolver,
    ContextSolver,
    IntegrationSolver,
)
from .validators import (
    CoherenceValidator,
    ContextIntegrityChecker,
    SystemStateValidator,
    ProcessValidator,
)
from .integrations import (
    MzekezekeIntegration,
    SpectacularIntegration,
    DiggidenIntegration,
    HatataIntegration,
    MachineryIntegration,
)
from .config import NicotineConfig
from .utils import (
    context_utils,
    puzzle_utils,
    validation_utils,
    scheduling_utils,
)

__version__ = "0.1.0"
__author__ = "Machinery Team"
__email__ = "team@machinery.dev"

__all__ = [
    # Core engine
    "NicotineEngine",
    "ContextState",
    "CoherenceLevel",
    
    # Puzzle generation and types
    "PuzzleGenerator",
    "ContextPuzzle",
    "LogicPuzzle",
    "MemoryPuzzle",
    "SummaryPuzzle",
    "IntegrationPuzzle",
    
    # Context management
    "ContextTracker",
    "ContextSummarizer",
    "ContextValidator",
    "ContextRefresher",
    
    # Scheduling systems
    "BreakScheduler",
    "AdaptiveScheduler",
    "ProcessCountScheduler",
    "TimeBasedScheduler",
    
    # Solving engines
    "PuzzleSolver",
    "LogicSolver",
    "MemorySolver",
    "ContextSolver",
    "IntegrationSolver",
    
    # Validation systems
    "CoherenceValidator",
    "ContextIntegrityChecker",
    "SystemStateValidator",
    "ProcessValidator",
    
    # Integrations
    "MzekezekeIntegration",
    "SpectacularIntegration",
    "DiggidenIntegration",
    "HatataIntegration",
    "MachineryIntegration",
    
    # Configuration
    "NicotineConfig",
    
    # Utilities
    "context_utils",
    "puzzle_utils",
    "validation_utils",
    "scheduling_utils",
] 
"""
Hatata: Stochastic Health Decision System

A decision-theoretic framework that uses Markov Decision Processes and stochastic 
methods to optimize health outcomes under uncertainty. Hatata provides:

- Markov Decision Process modeling for health state transitions
- Utility function optimization across different goal states
- Stochastic evidence integration from mzekezeke and diggiden
- Policy optimization for health interventions
- Probabilistic reasoning under uncertainty
- Multi-objective decision making for complex health scenarios

Core Philosophy:
Health decisions involve navigating uncertainty, trade-offs, and multiple competing
objectives. By modeling health as a stochastic process with clearly defined states,
actions, and utilities, we can make optimal decisions even when faced with
incomplete information and adversarial challenges.

The name "hatata" (Swahili for "step by step") reflects the sequential decision-making
nature of health optimization, where each step builds upon previous states and
influences future possibilities.
"""

__version__ = "0.1.0"
__author__ = "Machinery Team"

# Core MDP and stochastic components
from .mdp_engine import MDPEngine, HealthMDP
from .state_manager import StateManager, HealthState
from .utility_optimizer import UtilityOptimizer, UtilityFunction
from .stochastic_processor import StochasticProcessor, StochasticEvidence

# Decision and policy components
from .decision_engine import DecisionEngine, PolicyOptimizer
from .evidence_integrator import EvidenceIntegrator, MultiSourceEvidence

# Integration interfaces
from .hatata_core import HatataCore
from .system_integration import SystemIntegration

__all__ = [
    # Core engine
    "HatataCore",
    "SystemIntegration",
    
    # MDP and state management
    "MDPEngine",
    "HealthMDP",
    "StateManager", 
    "HealthState",
    
    # Utility and optimization
    "UtilityOptimizer",
    "UtilityFunction",
    "PolicyOptimizer",
    
    # Stochastic processing
    "StochasticProcessor",
    "StochasticEvidence",
    
    # Decision making
    "DecisionEngine",
    
    # Evidence integration
    "EvidenceIntegrator",
    "MultiSourceEvidence",
] 
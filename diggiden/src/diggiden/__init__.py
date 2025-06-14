"""
Diggiden: Adversarial Health Challenge System

An antagonist system that challenges health optimization frameworks by:
- Simulating disease processes and pathogenic challenges
- Finding vulnerabilities in health predictions and interventions
- Stress testing health monitoring systems
- Providing adversarial training data for robust health AI
- Modeling the complex interplay of health degradation processes

Core Philosophy:
Health is maintained through constant struggle against entropy, pathogens, 
stress, and system degradation. By actively modeling these adversarial forces,
we can build more resilient and robust health optimization systems.

The name "diggiden" reflects the persistent, undermining nature of disease
and degradation processes that continuously challenge biological systems.
"""

__version__ = "0.1.0"
__author__ = "Machinery Team"

# Core adversarial components
from .adversarial_engine import AdversarialEngine
from .disease_simulation import DiseaseSimulator, PathogenModel
from .vulnerability_detection import VulnerabilityDetector, WeaknessAnalyzer
from .stress_testing import StressTester, SystemChallenges
from .pathogen_modeling import PathogenEvolutionModel, InfectionDynamics

# Integration interfaces
from .antagonist import HealthAntagonist
from .challenge_protocols import ChallengeProtocol, AdversarialScenario

__all__ = [
    # Core engine
    "AdversarialEngine",
    "HealthAntagonist",
    
    # Disease and pathogen simulation
    "DiseaseSimulator",
    "PathogenModel", 
    "PathogenEvolutionModel",
    "InfectionDynamics",
    
    # Vulnerability and weakness detection
    "VulnerabilityDetector",
    "WeaknessAnalyzer",
    
    # Stress testing
    "StressTester",
    "SystemChallenges",
    
    # Challenge protocols
    "ChallengeProtocol",
    "AdversarialScenario",
] 
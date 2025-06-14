"""
Health Antagonist - The primary adversary to health optimization systems

This module embodies your key insight: health is a complex mixture where different 
systems operate at different percentages (100%, 90%, etc.) and a person can still 
feel well. The antagonist challenges this balance by:

1. Testing system resilience when multiple systems are sub-optimal
2. Finding the breaking points where the mixture becomes unstable
3. Simulating real-world conditions where perfect health is impossible
4. Forcing the health system to optimize for robustness, not perfection
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Represents the state of a biological system with percentage functionality."""
    system_name: str
    functionality_percentage: float  # 0.0 to 100.0
    reserve_capacity: float
    stress_tolerance: float
    compensation_ability: float  # How well it can help other systems
    degradation_rate: float
    recovery_potential: float


class HealthBalance:
    """
    Represents the complex mixture of system states that constitute health.
    
    Key insight: A person can feel "well" even when no single system is at 100%,
    as long as the overall balance is maintained and critical thresholds aren't crossed.
    """
    
    def __init__(self):
        self.systems: Dict[str, SystemState] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        self.global_wellness_threshold: float = 65.0  # Overall threshold for feeling "well"
        self.critical_system_threshold: float = 30.0  # Below this, system failure
        
    def add_system(self, system: SystemState) -> None:
        """Add a biological system to the balance."""
        self.systems[system.system_name] = system
        
    def calculate_wellness_score(self) -> float:
        """
        Calculate overall wellness based on system balance and interactions.
        
        This is where the magic happens - the complex mixture calculation.
        """
        if not self.systems:
            return 0.0
        
        # Base score from individual systems
        individual_scores = []
        for system in self.systems.values():
            # Systems below critical threshold contribute negatively
            if system.functionality_percentage < self.critical_system_threshold:
                individual_scores.append(system.functionality_percentage * 0.5)  # Penalty
            else:
                individual_scores.append(system.functionality_percentage)
        
        base_score = np.mean(individual_scores)
        
        # Interaction bonuses/penalties
        interaction_modifier = self._calculate_system_interactions()
        
        # Compensation effects - strong systems helping weak ones
        compensation_bonus = self._calculate_compensation_effects()
        
        # Reserve capacity buffer
        reserve_bonus = self._calculate_reserve_buffer()
        
        final_score = base_score + interaction_modifier + compensation_bonus + reserve_bonus
        return max(0.0, min(100.0, final_score))
    
    def _calculate_system_interactions(self) -> float:
        """Calculate how system interactions affect overall wellness."""
        if len(self.systems) < 2:
            return 0.0
        
        interaction_effects = []
        system_names = list(self.systems.keys())
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                # Get interaction strength
                interaction_key = tuple(sorted([sys1, sys2]))
                interaction_strength = self.interaction_matrix.get(interaction_key, 0.5)
                
                # Calculate synergy or interference
                sys1_func = self.systems[sys1].functionality_percentage
                sys2_func = self.systems[sys2].functionality_percentage
                
                # Positive interaction when both systems are functional
                if sys1_func > 70 and sys2_func > 70:
                    interaction_effects.append(interaction_strength * 2.0)  # Synergy
                elif sys1_func < 40 or sys2_func < 40:
                    interaction_effects.append(-interaction_strength * 1.5)  # Interference
                else:
                    interaction_effects.append(0.0)  # Neutral
        
        return np.mean(interaction_effects) if interaction_effects else 0.0
    
    def _calculate_compensation_effects(self) -> float:
        """Calculate how strong systems compensate for weak ones."""
        compensation_effects = []
        
        for system in self.systems.values():
            if system.functionality_percentage > 80:  # Strong system
                # Find weak systems it can help
                for other_system in self.systems.values():
                    if (other_system.system_name != system.system_name and 
                        other_system.functionality_percentage < 60):
                        
                        # Compensation effect
                        compensation_power = system.compensation_ability * system.reserve_capacity
                        weakness_severity = (60 - other_system.functionality_percentage) / 60
                        compensation_effect = compensation_power * weakness_severity * 0.3
                        compensation_effects.append(compensation_effect)
        
        return sum(compensation_effects)
    
    def _calculate_reserve_buffer(self) -> float:
        """Calculate wellness buffer from reserve capacities."""
        reserve_scores = [system.reserve_capacity for system in self.systems.values()]
        average_reserve = np.mean(reserve_scores) if reserve_scores else 0.0
        return average_reserve * 0.2  # Reserve provides small but important buffer
    
    def is_person_feeling_well(self) -> bool:
        """Determine if person feels well based on complex system balance."""
        wellness_score = self.calculate_wellness_score()
        
        # Additional checks beyond simple score
        critical_failures = sum(
            1 for system in self.systems.values() 
            if system.functionality_percentage < self.critical_system_threshold
        )
        
        # Person feels well if:
        # 1. Overall wellness above threshold
        # 2. No more than one critical system failure
        # 3. At least one system above 80% (providing leadership)
        
        has_strong_system = any(
            system.functionality_percentage > 80 
            for system in self.systems.values()
        )
        
        return (wellness_score > self.global_wellness_threshold and 
                critical_failures <= 1 and 
                has_strong_system)


class HealthAntagonist:
    """
    The primary antagonist that challenges health optimization systems.
    
    This antagonist embodies the constant forces working against health:
    - Entropy and aging
    - Environmental stressors
    - Pathogens and toxins
    - Resource limitations
    - System conflicts and trade-offs
    """
    
    def __init__(
        self,
        target_health_system: Any,
        antagonist_intensity: float = 0.6,
        learning_enabled: bool = True
    ):
        self.target_system = target_health_system
        self.intensity = antagonist_intensity
        self.learning_enabled = learning_enabled
        
        # Current health balance being challenged
        self.current_balance = HealthBalance()
        self._initialize_typical_systems()
        
        # Antagonist knowledge and adaptation
        self.discovered_vulnerabilities: Dict[str, float] = {}
        self.successful_attack_patterns: List[Dict[str, Any]] = []
        self.system_resilience_map: Dict[str, float] = {}
        
        # Challenge history
        self.challenges_issued: List[Dict[str, Any]] = []
        self.victories: int = 0
        self.defeats: int = 0
        
        logger.info(f"HealthAntagonist initialized with intensity {antagonist_intensity}")
    
    def _initialize_typical_systems(self) -> None:
        """Initialize typical biological systems with realistic starting values."""
        typical_systems = [
            SystemState(
                system_name="cardiovascular",
                functionality_percentage=85.0,
                reserve_capacity=0.6,
                stress_tolerance=0.7,
                compensation_ability=0.8,
                degradation_rate=0.02,
                recovery_potential=0.7
            ),
            SystemState(
                system_name="immune",
                functionality_percentage=75.0,
                reserve_capacity=0.5,
                stress_tolerance=0.6,
                compensation_ability=0.4,
                degradation_rate=0.03,
                recovery_potential=0.8
            ),
            SystemState(
                system_name="nervous",
                functionality_percentage=90.0,
                reserve_capacity=0.7,
                stress_tolerance=0.5,
                compensation_ability=0.9,
                degradation_rate=0.015,
                recovery_potential=0.6
            ),
            SystemState(
                system_name="metabolic",
                functionality_percentage=80.0,
                reserve_capacity=0.4,
                stress_tolerance=0.8,
                compensation_ability=0.7,
                degradation_rate=0.025,
                recovery_potential=0.9
            ),
            SystemState(
                system_name="respiratory",
                functionality_percentage=88.0,
                reserve_capacity=0.8,
                stress_tolerance=0.9,
                compensation_ability=0.5,
                degradation_rate=0.02,
                recovery_potential=0.7
            )
        ]
        
        for system in typical_systems:
            self.current_balance.add_system(system)
        
        # Set up system interactions
        self.current_balance.interaction_matrix = {
            ("cardiovascular", "respiratory"): 0.9,  # Strong positive interaction
            ("cardiovascular", "metabolic"): 0.8,
            ("immune", "nervous"): 0.6,
            ("metabolic", "nervous"): 0.7,
            ("immune", "metabolic"): 0.5
        }
    
    def challenge_system_balance(
        self,
        challenge_type: str = "mixed_degradation",
        target_systems: Optional[List[str]] = None,
        challenge_duration: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        Issue a challenge to the health balance.
        
        This is where we test your insight - can the health system maintain
        wellness when we degrade multiple systems to different percentages?
        """
        if target_systems is None:
            # Select 2-3 systems to challenge
            available_systems = list(self.current_balance.systems.keys())
            target_systems = random.sample(available_systems, random.randint(2, 3))
        
        # Record pre-challenge state
        pre_challenge_wellness = self.current_balance.calculate_wellness_score()
        pre_challenge_feeling_well = self.current_balance.is_person_feeling_well()
        
        # Apply the challenge
        challenge_effects = self._apply_challenge(challenge_type, target_systems)
        
        # Calculate post-challenge state
        post_challenge_wellness = self.current_balance.calculate_wellness_score()
        post_challenge_feeling_well = self.current_balance.is_person_feeling_well()
        
        # Determine challenge success
        challenge_successful = (
            not post_challenge_feeling_well or  # Person no longer feels well
            (pre_challenge_wellness - post_challenge_wellness) > 20  # Significant degradation
        )
        
        challenge_record = {
            "challenge_id": f"challenge_{datetime.now().isoformat()}",
            "challenge_type": challenge_type,
            "target_systems": target_systems,
            "challenge_effects": challenge_effects,
            "pre_wellness_score": pre_challenge_wellness,
            "post_wellness_score": post_challenge_wellness,
            "pre_feeling_well": pre_challenge_feeling_well,
            "post_feeling_well": post_challenge_feeling_well,
            "challenge_successful": challenge_successful,
            "wellness_drop": pre_challenge_wellness - post_challenge_wellness,
            "timestamp": datetime.now(),
            "duration": challenge_duration
        }
        
        # Update statistics
        if challenge_successful:
            self.victories += 1
        else:
            self.defeats += 1
        
        self.challenges_issued.append(challenge_record)
        
        # Learn from the outcome
        if self.learning_enabled:
            self._learn_from_challenge(challenge_record)
        
        logger.info(
            f"Challenge {challenge_type} on {target_systems}: "
            f"Success={challenge_successful}, "
            f"Wellness: {pre_challenge_wellness:.1f} → {post_challenge_wellness:.1f}"
        )
        
        return challenge_record
    
    def _apply_challenge(
        self, 
        challenge_type: str, 
        target_systems: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Apply the challenge to specific systems."""
        effects = {}
        
        for system_name in target_systems:
            if system_name not in self.current_balance.systems:
                continue
            
            system = self.current_balance.systems[system_name]
            original_functionality = system.functionality_percentage
            
            # Apply different types of challenges
            if challenge_type == "gradual_degradation":
                # Reduce functionality by 10-30%
                reduction = random.uniform(10, 30)
                system.functionality_percentage = max(20, system.functionality_percentage - reduction)
                
            elif challenge_type == "stress_overload":
                # Reduce functionality based on stress tolerance
                stress_impact = (1.0 - system.stress_tolerance) * random.uniform(15, 40)
                system.functionality_percentage = max(25, system.functionality_percentage - stress_impact)
                
            elif challenge_type == "resource_depletion":
                # Affect reserve capacity and functionality
                system.reserve_capacity *= random.uniform(0.5, 0.8)
                reduction = random.uniform(8, 25)
                system.functionality_percentage = max(30, system.functionality_percentage - reduction)
                
            elif challenge_type == "mixed_degradation":
                # Multiple small hits across different aspects
                functionality_hit = random.uniform(5, 20)
                reserve_hit = random.uniform(0.1, 0.3)
                
                system.functionality_percentage = max(25, system.functionality_percentage - functionality_hit)
                system.reserve_capacity = max(0.1, system.reserve_capacity - reserve_hit)
            
            effects[system_name] = {
                "original_functionality": original_functionality,
                "new_functionality": system.functionality_percentage,
                "functionality_change": system.functionality_percentage - original_functionality,
                "challenge_type": challenge_type
            }
        
        return effects
    
    def _learn_from_challenge(self, challenge_record: Dict[str, Any]) -> None:
        """Learn from challenge outcomes to improve future attacks."""
        target_systems = challenge_record["target_systems"]
        challenge_successful = challenge_record["challenge_successful"]
        wellness_drop = challenge_record["wellness_drop"]
        
        # Update vulnerability knowledge
        for system in target_systems:
            if system not in self.discovered_vulnerabilities:
                self.discovered_vulnerabilities[system] = 0.5
            
            if challenge_successful:
                # This system showed vulnerability
                self.discovered_vulnerabilities[system] = min(
                    1.0, 
                    self.discovered_vulnerabilities[system] + 0.1
                )
            else:
                # System showed resilience
                self.discovered_vulnerabilities[system] = max(
                    0.1,
                    self.discovered_vulnerabilities[system] - 0.05
                )
        
        # Record successful attack patterns
        if challenge_successful and wellness_drop > 15:
            self.successful_attack_patterns.append({
                "challenge_type": challenge_record["challenge_type"],
                "target_systems": target_systems,
                "wellness_drop": wellness_drop,
                "timestamp": challenge_record["timestamp"]
            })
    
    def get_current_balance_report(self) -> Dict[str, Any]:
        """Get detailed report of current health balance."""
        system_states = {}
        for name, system in self.current_balance.systems.items():
            system_states[name] = {
                "functionality_percentage": system.functionality_percentage,
                "reserve_capacity": system.reserve_capacity,
                "stress_tolerance": system.stress_tolerance,
                "status": self._classify_system_status(system.functionality_percentage)
            }
        
        wellness_score = self.current_balance.calculate_wellness_score()
        feeling_well = self.current_balance.is_person_feeling_well()
        
        return {
            "overall_wellness_score": wellness_score,
            "person_feeling_well": feeling_well,
            "system_states": system_states,
            "wellness_classification": self._classify_wellness(wellness_score),
            "critical_systems": [
                name for name, system in self.current_balance.systems.items()
                if system.functionality_percentage < 40
            ],
            "strong_systems": [
                name for name, system in self.current_balance.systems.items()
                if system.functionality_percentage > 80
            ],
            "balance_analysis": self._analyze_balance_stability()
        }
    
    def _classify_system_status(self, functionality: float) -> str:
        """Classify individual system status."""
        if functionality >= 90:
            return "excellent"
        elif functionality >= 80:
            return "good"
        elif functionality >= 70:
            return "adequate"
        elif functionality >= 50:
            return "compromised"
        elif functionality >= 30:
            return "failing"
        else:
            return "critical"
    
    def _classify_wellness(self, wellness_score: float) -> str:
        """Classify overall wellness."""
        if wellness_score >= 85:
            return "thriving"
        elif wellness_score >= 70:
            return "healthy"
        elif wellness_score >= 55:
            return "managing"
        elif wellness_score >= 40:
            return "struggling"
        else:
            return "crisis"
    
    def _analyze_balance_stability(self) -> Dict[str, Any]:
        """Analyze the stability of the current balance."""
        functionalities = [
            system.functionality_percentage 
            for system in self.current_balance.systems.values()
        ]
        
        balance_stability = {
            "variance": float(np.var(functionalities)),
            "range": float(max(functionalities) - min(functionalities)),
            "systems_below_60": sum(1 for f in functionalities if f < 60),
            "systems_above_80": sum(1 for f in functionalities if f > 80),
            "stability_rating": "stable" if np.var(functionalities) < 200 else "unstable"
        }
        
        return balance_stability
    
    def get_antagonist_performance(self) -> Dict[str, Any]:
        """Get performance statistics of the antagonist."""
        total_challenges = len(self.challenges_issued)
        success_rate = self.victories / total_challenges if total_challenges > 0 else 0.0
        
        return {
            "total_challenges_issued": total_challenges,
            "victories": self.victories,
            "defeats": self.defeats,
            "success_rate": success_rate,
            "discovered_vulnerabilities": self.discovered_vulnerabilities,
            "successful_attack_patterns": len(self.successful_attack_patterns),
            "learning_enabled": self.learning_enabled,
            "current_intensity": self.intensity,
            "performance_rating": "effective" if success_rate > 0.4 else "needs_improvement"
        }
    
    def simulate_health_deterioration_scenario(
        self, 
        scenario_name: str = "aging_simulation",
        duration_days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Simulate a long-term health deterioration scenario.
        
        This demonstrates your insight about how health is maintained
        despite gradual degradation of multiple systems.
        """
        scenario_log = []
        
        # Simulate gradual changes over time
        for day in range(0, duration_days, 30):  # Monthly checkpoints
            # Apply small degradations
            for system in self.current_balance.systems.values():
                # Natural degradation
                natural_decline = system.degradation_rate * 30  # Monthly decline
                system.functionality_percentage = max(
                    20, 
                    system.functionality_percentage - natural_decline
                )
                
                # Random small stressors
                if random.random() < 0.3:  # 30% chance of stress each month
                    stress_impact = random.uniform(1, 5)
                    system.functionality_percentage = max(
                        20,
                        system.functionality_percentage - stress_impact
                    )
                
                # Occasional recovery
                if random.random() < 0.2:  # 20% chance of recovery
                    recovery = system.recovery_potential * random.uniform(2, 8)
                    system.functionality_percentage = min(
                        100,
                        system.functionality_percentage + recovery
                    )
            
            # Record state
            wellness_score = self.current_balance.calculate_wellness_score()
            feeling_well = self.current_balance.is_person_feeling_well()
            
            scenario_log.append({
                "day": day,
                "wellness_score": wellness_score,
                "feeling_well": feeling_well,
                "system_states": {
                    name: system.functionality_percentage
                    for name, system in self.current_balance.systems.items()
                }
            })
        
        return scenario_log 
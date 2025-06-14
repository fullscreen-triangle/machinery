"""
Markov Decision Process Engine for Health State Modeling

This module implements the core MDP framework for health decision making,
including state spaces, action spaces, transition probabilities, and reward functions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
from scipy.special import softmax
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class HealthStateCategory(Enum):
    """Categories of health states for MDP modeling."""
    OPTIMAL = "optimal"              # Peak health state
    EXCELLENT = "excellent"          # Very good health
    GOOD = "good"                   # Good health
    ADEQUATE = "adequate"           # Adequate health
    COMPROMISED = "compromised"     # Compromised health
    FAILING = "failing"             # Failing health
    CRITICAL = "critical"           # Critical health state


@dataclass
class HealthState:
    """
    Represents a health state in the MDP.
    
    Each state captures:
    - Physical health metrics (cardiovascular, immune, etc.)
    - Mental/cognitive state
    - Environmental factors
    - Resource availability
    - Risk factors
    """
    state_id: str
    category: HealthStateCategory
    
    # System functionality levels (0.0 to 1.0)
    cardiovascular: float = 0.8
    immune: float = 0.8
    nervous: float = 0.8
    metabolic: float = 0.8
    respiratory: float = 0.8
    
    # Derived metrics
    overall_wellness: float = 0.8
    stress_level: float = 0.3
    energy_level: float = 0.7
    recovery_capacity: float = 0.6
    
    # Context factors
    age_factor: float = 1.0
    environmental_stress: float = 0.2
    resource_availability: float = 0.8
    social_support: float = 0.7
    
    # Temporal aspects
    state_duration: float = 1.0  # How long in this state
    trajectory_trend: float = 0.0  # -1 (declining) to +1 (improving)
    
    # Uncertainty measures
    state_confidence: float = 0.8
    measurement_noise: float = 0.1
    
    def __post_init__(self):
        self.overall_wellness = self._calculate_overall_wellness()
    
    def _calculate_overall_wellness(self) -> float:
        """Calculate overall wellness from system components."""
        system_scores = [
            self.cardiovascular, self.immune, self.nervous,
            self.metabolic, self.respiratory
        ]
        
        # Weighted average with contextual adjustments
        base_wellness = np.mean(system_scores)
        
        # Apply context adjustments
        context_adjustment = (
            self.age_factor * 0.1 +
            (1.0 - self.environmental_stress) * 0.1 +
            self.resource_availability * 0.1 +
            self.social_support * 0.1 +
            (1.0 - self.stress_level) * 0.2
        )
        
        return min(1.0, max(0.0, base_wellness + context_adjustment))
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for ML processing."""
        return np.array([
            self.cardiovascular, self.immune, self.nervous,
            self.metabolic, self.respiratory, self.overall_wellness,
            self.stress_level, self.energy_level, self.recovery_capacity,
            self.age_factor, self.environmental_stress,
            self.resource_availability, self.social_support,
            self.state_duration, self.trajectory_trend,
            self.state_confidence, self.measurement_noise
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, state_id: str = None) -> 'HealthState':
        """Create HealthState from numerical vector."""
        if len(vector) < 17:
            raise ValueError("Vector must have at least 17 elements")
        
        # Determine category based on overall wellness
        overall_wellness = vector[5]
        if overall_wellness >= 0.9:
            category = HealthStateCategory.OPTIMAL
        elif overall_wellness >= 0.8:
            category = HealthStateCategory.EXCELLENT
        elif overall_wellness >= 0.7:
            category = HealthStateCategory.GOOD
        elif overall_wellness >= 0.6:
            category = HealthStateCategory.ADEQUATE
        elif overall_wellness >= 0.4:
            category = HealthStateCategory.COMPROMISED
        elif overall_wellness >= 0.2:
            category = HealthStateCategory.FAILING
        else:
            category = HealthStateCategory.CRITICAL
        
        return cls(
            state_id=state_id or f"state_{datetime.now().isoformat()}",
            category=category,
            cardiovascular=vector[0],
            immune=vector[1],
            nervous=vector[2],
            metabolic=vector[3],
            respiratory=vector[4],
            overall_wellness=vector[5],
            stress_level=vector[6],
            energy_level=vector[7],
            recovery_capacity=vector[8],
            age_factor=vector[9],
            environmental_stress=vector[10],
            resource_availability=vector[11],
            social_support=vector[12],
            state_duration=vector[13],
            trajectory_trend=vector[14],
            state_confidence=vector[15],
            measurement_noise=vector[16] if len(vector) > 16 else 0.1
        )


@dataclass
class HealthAction:
    """
    Represents an action/intervention in the health MDP.
    """
    action_id: str
    action_type: str  # "medical", "lifestyle", "preventive", "emergency"
    
    # Action characteristics
    intensity: float = 0.5  # 0.0 to 1.0
    duration: float = 1.0   # Duration in time units
    cost: float = 0.1       # Resource cost
    risk: float = 0.05      # Risk level
    efficacy: float = 0.7   # Expected efficacy
    
    # Target systems
    target_systems: List[str] = field(default_factory=list)
    
    # Side effects and contraindications  
    side_effects: Dict[str, float] = field(default_factory=dict)
    contraindications: List[str] = field(default_factory=list)
    
    # Temporal aspects
    onset_time: float = 0.1     # Time to effect
    peak_time: float = 0.5      # Time to peak effect
    decay_time: float = 2.0     # Effect decay time
    
    def is_applicable(self, state: HealthState) -> bool:
        """Check if action is applicable in given state."""
        # Check contraindications
        if state.category.value in self.contraindications:
            return False
        
        # Check resource requirements
        if self.cost > state.resource_availability:
            return False
        
        # Check if state is too critical for non-emergency actions
        if (state.category == HealthStateCategory.CRITICAL and 
            self.action_type != "emergency"):
            return False
        
        return True


class HealthMDP:
    """
    Markov Decision Process model for health state transitions.
    
    This class implements a complete MDP framework including:
    - State space definition
    - Action space definition  
    - Transition probability functions
    - Reward/utility functions
    - Policy optimization
    """
    
    def __init__(
        self,
        state_discretization: int = 50,
        time_horizon: int = 100,
        discount_factor: float = 0.95,
        uncertainty_level: float = 0.1
    ):
        self.state_discretization = state_discretization
        self.time_horizon = time_horizon
        self.discount_factor = discount_factor
        self.uncertainty_level = uncertainty_level
        
        # MDP components
        self.states: Dict[str, HealthState] = {}
        self.actions: Dict[str, HealthAction] = {}
        self.transition_probabilities: Dict[Tuple[str, str, str], float] = {}
        self.rewards: Dict[Tuple[str, str], float] = {}
        
        # Policy and value functions
        self.policy: Dict[str, str] = {}  # state_id -> action_id
        self.value_function: Dict[str, float] = {}
        self.q_function: Dict[Tuple[str, str], float] = {}
        
        # State transition graph
        self.transition_graph = nx.DiGraph()
        
        # Learning and optimization
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        
        logger.info(f"HealthMDP initialized with {state_discretization} states")
        
        # Initialize default states and actions
        self._initialize_default_states()
        self._initialize_default_actions()
    
    def _initialize_default_states(self) -> None:
        """Initialize a set of representative health states."""
        # Create states across the health spectrum
        state_templates = [
            # Optimal health
            {
                "state_id": "optimal_health",
                "category": HealthStateCategory.OPTIMAL,
                "cardiovascular": 0.95, "immune": 0.9, "nervous": 0.95,
                "metabolic": 0.9, "respiratory": 0.95,
                "stress_level": 0.1, "energy_level": 0.9
            },
            # Excellent health
            {
                "state_id": "excellent_health", 
                "category": HealthStateCategory.EXCELLENT,
                "cardiovascular": 0.85, "immune": 0.8, "nervous": 0.85,
                "metabolic": 0.8, "respiratory": 0.85,
                "stress_level": 0.2, "energy_level": 0.8
            },
            # Good health
            {
                "state_id": "good_health",
                "category": HealthStateCategory.GOOD,
                "cardiovascular": 0.75, "immune": 0.7, "nervous": 0.75,
                "metabolic": 0.7, "respiratory": 0.75,
                "stress_level": 0.3, "energy_level": 0.7
            },
            # Adequate health
            {
                "state_id": "adequate_health",
                "category": HealthStateCategory.ADEQUATE,
                "cardiovascular": 0.65, "immune": 0.6, "nervous": 0.65,
                "metabolic": 0.6, "respiratory": 0.65,
                "stress_level": 0.4, "energy_level": 0.6
            },
            # Compromised health
            {
                "state_id": "compromised_health",
                "category": HealthStateCategory.COMPROMISED,
                "cardiovascular": 0.5, "immune": 0.45, "nervous": 0.5,
                "metabolic": 0.45, "respiratory": 0.5,
                "stress_level": 0.6, "energy_level": 0.4
            },
            # Failing health
            {
                "state_id": "failing_health",
                "category": HealthStateCategory.FAILING,
                "cardiovascular": 0.35, "immune": 0.3, "nervous": 0.35,
                "metabolic": 0.3, "respiratory": 0.35,
                "stress_level": 0.7, "energy_level": 0.3
            },
            # Critical health
            {
                "state_id": "critical_health",
                "category": HealthStateCategory.CRITICAL,
                "cardiovascular": 0.2, "immune": 0.15, "nervous": 0.2,
                "metabolic": 0.15, "respiratory": 0.2,
                "stress_level": 0.9, "energy_level": 0.1
            }
        ]
        
        for template in state_templates:
            state = HealthState(**template)
            self.add_state(state)
    
    def _initialize_default_actions(self) -> None:
        """Initialize a set of representative health actions."""
        action_templates = [
            # Lifestyle interventions
            {
                "action_id": "exercise_moderate",
                "action_type": "lifestyle",
                "intensity": 0.6, "duration": 1.0, "cost": 0.1,
                "efficacy": 0.7, "target_systems": ["cardiovascular", "metabolic"]
            },
            {
                "action_id": "exercise_intense",
                "action_type": "lifestyle", 
                "intensity": 0.9, "duration": 1.0, "cost": 0.2,
                "efficacy": 0.8, "target_systems": ["cardiovascular", "respiratory"],
                "contraindications": ["critical"]
            },
            {
                "action_id": "stress_reduction",
                "action_type": "lifestyle",
                "intensity": 0.5, "duration": 1.0, "cost": 0.05,
                "efficacy": 0.6, "target_systems": ["nervous"]
            },
            {
                "action_id": "nutrition_optimization",
                "action_type": "lifestyle",
                "intensity": 0.4, "duration": 2.0, "cost": 0.15,
                "efficacy": 0.65, "target_systems": ["metabolic", "immune"]
            },
            
            # Medical interventions
            {
                "action_id": "preventive_care",
                "action_type": "preventive",
                "intensity": 0.3, "duration": 0.5, "cost": 0.2,
                "efficacy": 0.5, "target_systems": ["immune"]
            },
            {
                "action_id": "medical_treatment",
                "action_type": "medical",
                "intensity": 0.8, "duration": 1.0, "cost": 0.5,
                "efficacy": 0.85, "risk": 0.1
            },
            {
                "action_id": "emergency_intervention",
                "action_type": "emergency",
                "intensity": 1.0, "duration": 0.5, "cost": 0.8,
                "efficacy": 0.9, "risk": 0.2,
                "onset_time": 0.05
            },
            
            # Recovery and rest
            {
                "action_id": "rest_recovery",
                "action_type": "lifestyle",
                "intensity": 0.2, "duration": 2.0, "cost": 0.05,
                "efficacy": 0.4, "target_systems": ["nervous"]
            },
            
            # No action (baseline)
            {
                "action_id": "no_action",
                "action_type": "baseline",
                "intensity": 0.0, "duration": 1.0, "cost": 0.0,
                "efficacy": 0.0
            }
        ]
        
        for template in action_templates:
            action = HealthAction(**template)
            self.add_action(action)
    
    def add_state(self, state: HealthState) -> None:
        """Add a state to the MDP."""
        self.states[state.state_id] = state
        self.transition_graph.add_node(state.state_id, state=state)
        
        # Initialize value function
        self.value_function[state.state_id] = 0.0
    
    def add_action(self, action: HealthAction) -> None:
        """Add an action to the MDP."""
        self.actions[action.action_id] = action
    
    def set_transition_probability(
        self, 
        from_state: str, 
        action: str, 
        to_state: str, 
        probability: float
    ) -> None:
        """Set transition probability P(s'|s,a)."""
        self.transition_probabilities[(from_state, action, to_state)] = probability
        
        # Add edge to transition graph
        if self.transition_graph.has_edge(from_state, to_state):
            self.transition_graph[from_state][to_state]['transitions'][action] = probability
        else:
            self.transition_graph.add_edge(
                from_state, to_state, 
                transitions={action: probability}
            )
    
    def get_transition_probability(
        self, 
        from_state: str, 
        action: str, 
        to_state: str
    ) -> float:
        """Get transition probability P(s'|s,a)."""
        return self.transition_probabilities.get((from_state, action, to_state), 0.0)
    
    def set_reward(self, state: str, action: str, reward: float) -> None:
        """Set reward R(s,a)."""
        self.rewards[(state, action)] = reward
    
    def get_reward(self, state: str, action: str) -> float:
        """Get reward R(s,a)."""
        return self.rewards.get((state, action), 0.0)
    
    def calculate_state_transition_probabilities(
        self, 
        from_state: HealthState, 
        action: HealthAction
    ) -> Dict[str, float]:
        """
        Calculate transition probabilities based on action effects and state dynamics.
        
        This uses a physics-based model of health state transitions.
        """
        transitions = {}
        
        # Base transition without action (natural progression)
        base_transitions = self._calculate_natural_transitions(from_state)
        
        # Action effects
        action_effects = self._calculate_action_effects(from_state, action)
        
        # Combine base transitions with action effects
        for to_state_id, base_prob in base_transitions.items():
            # Modify probability based on action effects
            if to_state_id in action_effects:
                effect_modifier = action_effects[to_state_id]
                modified_prob = base_prob * (1.0 + effect_modifier)
            else:
                modified_prob = base_prob
            
            # Add uncertainty
            noise = np.random.normal(0, self.uncertainty_level)
            final_prob = max(0.0, min(1.0, modified_prob + noise))
            
            transitions[to_state_id] = final_prob
        
        # Normalize probabilities
        total_prob = sum(transitions.values())
        if total_prob > 0:
            transitions = {k: v / total_prob for k, v in transitions.items()}
        
        return transitions
    
    def _calculate_natural_transitions(self, state: HealthState) -> Dict[str, float]:
        """Calculate natural state transitions without interventions."""
        transitions = {}
        
        # Transition probabilities based on current state and trajectory
        current_wellness = state.overall_wellness
        trajectory = state.trajectory_trend
        
        # Default probabilities for staying in same category
        stay_prob = 0.7
        
        # Adjustment based on trajectory
        if trajectory > 0.1:  # Improving
            improve_prob = 0.2 + trajectory * 0.1
            decline_prob = 0.1 - trajectory * 0.05
        elif trajectory < -0.1:  # Declining
            improve_prob = 0.1 + trajectory * 0.05
            decline_prob = 0.2 - trajectory * 0.1
        else:  # Stable
            improve_prob = 0.15
            decline_prob = 0.15
        
        # Distribute probabilities among states
        current_category = state.category
        
        for state_id, target_state in self.states.items():
            if state_id == state.state_id:
                transitions[state_id] = stay_prob
            elif target_state.category.value == current_category.value:
                transitions[state_id] = stay_prob * 0.3  # Similar states
            elif self._is_better_state(target_state.category, current_category):
                transitions[state_id] = improve_prob / self._count_better_states(current_category)
            elif self._is_worse_state(target_state.category, current_category):
                transitions[state_id] = decline_prob / self._count_worse_states(current_category)
            else:
                transitions[state_id] = 0.01  # Small probability for distant states
        
        return transitions
    
    def _calculate_action_effects(
        self, 
        state: HealthState, 
        action: HealthAction
    ) -> Dict[str, float]:
        """Calculate how action modifies transition probabilities."""
        effects = {}
        
        if not action.is_applicable(state):
            return effects
        
        # Action efficacy determines strength of effects
        effect_strength = action.efficacy * action.intensity
        
        # Positive effects (move toward better states)
        for state_id, target_state in self.states.items():
            if self._is_better_state(target_state.category, state.category):
                # Action increases probability of improvement
                effects[state_id] = effect_strength * 0.5
            elif self._is_worse_state(target_state.category, state.category):
                # Action decreases probability of decline
                effects[state_id] = -effect_strength * 0.3
        
        # Side effects
        for system, side_effect_strength in action.side_effects.items():
            # Negative effects on specific systems
            for state_id, target_state in self.states.items():
                if (hasattr(target_state, system) and 
                    getattr(target_state, system) < getattr(state, system)):
                    effects[state_id] = effects.get(state_id, 0) + side_effect_strength
        
        return effects
    
    def _is_better_state(self, state1: HealthStateCategory, state2: HealthStateCategory) -> bool:
        """Check if state1 is better than state2."""
        category_order = [
            HealthStateCategory.CRITICAL,
            HealthStateCategory.FAILING,
            HealthStateCategory.COMPROMISED,
            HealthStateCategory.ADEQUATE,
            HealthStateCategory.GOOD,
            HealthStateCategory.EXCELLENT,
            HealthStateCategory.OPTIMAL
        ]
        return category_order.index(state1) > category_order.index(state2)
    
    def _is_worse_state(self, state1: HealthStateCategory, state2: HealthStateCategory) -> bool:
        """Check if state1 is worse than state2."""
        return self._is_better_state(state2, state1)
    
    def _count_better_states(self, category: HealthStateCategory) -> int:
        """Count number of better state categories."""
        return sum(1 for _, state in self.states.items() 
                  if self._is_better_state(state.category, category))
    
    def _count_worse_states(self, category: HealthStateCategory) -> int:
        """Count number of worse state categories."""
        return sum(1 for _, state in self.states.items() 
                  if self._is_worse_state(state.category, category))
    
    def calculate_utility_reward(self, state: HealthState, action: HealthAction) -> float:
        """Calculate utility-based reward for state-action pair."""
        # Base utility from state quality
        base_utility = state.overall_wellness * 100
        
        # Action costs
        action_cost = action.cost * 10
        
        # Risk penalty
        risk_penalty = action.risk * 20
        
        # Efficacy bonus for beneficial actions
        efficacy_bonus = 0
        if action.efficacy > 0.5:
            efficacy_bonus = (action.efficacy - 0.5) * 10
        
        # Time preference (longer duration actions have diminishing returns)
        time_discount = 1.0 / (1.0 + action.duration * 0.1)
        
        total_reward = (base_utility + efficacy_bonus - action_cost - risk_penalty) * time_discount
        
        return total_reward
    
    def solve_mdp(self, method: str = "value_iteration", max_iterations: int = 1000) -> None:
        """
        Solve the MDP to find optimal policy.
        
        Args:
            method: Solution method ("value_iteration", "policy_iteration", "q_learning")
            max_iterations: Maximum number of iterations
        """
        if method == "value_iteration":
            self._value_iteration(max_iterations)
        elif method == "policy_iteration":
            self._policy_iteration(max_iterations)
        elif method == "q_learning":
            self._q_learning(max_iterations)
        else:
            raise ValueError(f"Unknown MDP solution method: {method}")
    
    def _value_iteration(self, max_iterations: int) -> None:
        """Solve MDP using value iteration."""
        for iteration in range(max_iterations):
            delta = 0.0
            new_values = {}
            
            for state_id, state in self.states.items():
                # Find best action for this state
                best_value = float('-inf')
                
                for action_id, action in self.actions.items():
                    if not action.is_applicable(state):
                        continue
                    
                    # Calculate expected value for this action
                    expected_value = self.get_reward(state_id, action_id)
                    
                    # Add discounted future value
                    transitions = self.calculate_state_transition_probabilities(state, action)
                    for next_state_id, prob in transitions.items():
                        expected_value += (self.discount_factor * prob * 
                                         self.value_function.get(next_state_id, 0.0))
                    
                    if expected_value > best_value:
                        best_value = expected_value
                
                new_values[state_id] = best_value
                delta = max(delta, abs(best_value - self.value_function[state_id]))
            
            # Update value function
            self.value_function.update(new_values)
            
            # Check convergence
            if delta < 1e-6:
                logger.info(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        # Extract optimal policy
        self._extract_policy_from_values()
    
    def _extract_policy_from_values(self) -> None:
        """Extract optimal policy from value function."""
        for state_id, state in self.states.items():
            best_action = None
            best_value = float('-inf')
            
            for action_id, action in self.actions.items():
                if not action.is_applicable(state):
                    continue
                
                # Calculate expected value for this action
                expected_value = self.get_reward(state_id, action_id)
                
                transitions = self.calculate_state_transition_probabilities(state, action)
                for next_state_id, prob in transitions.items():
                    expected_value += (self.discount_factor * prob * 
                                     self.value_function.get(next_state_id, 0.0))
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action_id
            
            if best_action:
                self.policy[state_id] = best_action
    
    def get_optimal_action(self, state: HealthState) -> Optional[HealthAction]:
        """Get optimal action for given state according to current policy."""
        if state.state_id in self.policy:
            action_id = self.policy[state.state_id]
            return self.actions.get(action_id)
        return None
    
    def simulate_trajectory(
        self, 
        initial_state: HealthState, 
        num_steps: int = 10,
        use_optimal_policy: bool = True
    ) -> List[Tuple[HealthState, HealthAction, float]]:
        """
        Simulate a trajectory through the MDP.
        
        Returns:
            List of (state, action, reward) tuples
        """
        trajectory = []
        current_state = initial_state
        
        for step in range(num_steps):
            # Choose action
            if use_optimal_policy:
                action = self.get_optimal_action(current_state)
            else:
                # Random action
                applicable_actions = [
                    action for action in self.actions.values()
                    if action.is_applicable(current_state)
                ]
                action = np.random.choice(applicable_actions) if applicable_actions else None
            
            if action is None:
                break
            
            # Calculate reward
            reward = self.calculate_utility_reward(current_state, action)
            
            # Record step
            trajectory.append((current_state, action, reward))
            
            # Transition to next state
            transitions = self.calculate_state_transition_probabilities(current_state, action)
            if transitions:
                next_state_id = np.random.choice(
                    list(transitions.keys()),
                    p=list(transitions.values())
                )
                current_state = self.states[next_state_id]
            else:
                break
        
        return trajectory
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about the MDP states."""
        stats = {
            "total_states": len(self.states),
            "state_categories": {},
            "average_wellness": 0.0,
            "wellness_distribution": []
        }
        
        wellness_scores = []
        for state in self.states.values():
            category = state.category.value
            stats["state_categories"][category] = stats["state_categories"].get(category, 0) + 1
            wellness_scores.append(state.overall_wellness)
        
        stats["average_wellness"] = np.mean(wellness_scores)
        stats["wellness_distribution"] = {
            "min": np.min(wellness_scores),
            "max": np.max(wellness_scores),
            "std": np.std(wellness_scores),
            "percentiles": {
                "25": np.percentile(wellness_scores, 25),
                "50": np.percentile(wellness_scores, 50),
                "75": np.percentile(wellness_scores, 75)
            }
        }
        
        return stats


class MDPEngine:
    """
    High-level engine for managing health MDPs and integration with other systems.
    """
    
    def __init__(self):
        self.mdp = HealthMDP()
        self.simulation_history: List[Dict[str, Any]] = []
        
        # Initialize MDP with calculated rewards
        self._initialize_mdp_rewards()
        
        logger.info("MDPEngine initialized")
    
    def _initialize_mdp_rewards(self) -> None:
        """Initialize reward function for all state-action pairs."""
        for state_id, state in self.mdp.states.items():
            for action_id, action in self.mdp.actions.items():
                if action.is_applicable(state):
                    reward = self.mdp.calculate_utility_reward(state, action)
                    self.mdp.set_reward(state_id, action_id, reward)
    
    def optimize_policy(self, method: str = "value_iteration") -> Dict[str, Any]:
        """Optimize MDP policy and return optimization results."""
        start_time = datetime.now()
        
        # Solve MDP
        self.mdp.solve_mdp(method=method)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        policy_stats = self._analyze_policy()
        
        results = {
            "optimization_method": method,
            "optimization_time": optimization_time,
            "policy_stats": policy_stats,
            "value_function_stats": {
                "mean_value": np.mean(list(self.mdp.value_function.values())),
                "max_value": max(self.mdp.value_function.values()),
                "min_value": min(self.mdp.value_function.values())
            },
            "convergence": True  # Simplified for now
        }
        
        return results
    
    def _analyze_policy(self) -> Dict[str, Any]:
        """Analyze the current policy."""
        policy_stats = {
            "total_states_with_policy": len(self.mdp.policy),
            "action_distribution": {},
            "policy_by_state_category": {}
        }
        
        for state_id, action_id in self.mdp.policy.items():
            # Count action types
            action = self.mdp.actions[action_id]
            action_type = action.action_type
            policy_stats["action_distribution"][action_type] = (
                policy_stats["action_distribution"].get(action_type, 0) + 1
            )
            
            # Group by state category
            state = self.mdp.states[state_id]
            category = state.category.value
            if category not in policy_stats["policy_by_state_category"]:
                policy_stats["policy_by_state_category"][category] = {}
            policy_stats["policy_by_state_category"][category][action_id] = (
                policy_stats["policy_by_state_category"][category].get(action_id, 0) + 1
            )
        
        return policy_stats
    
    def recommend_action(self, current_state: HealthState) -> Tuple[HealthAction, float]:
        """
        Recommend optimal action for current state.
        
        Returns:
            Tuple of (recommended_action, expected_utility)
        """
        # Ensure state is in MDP
        if current_state.state_id not in self.mdp.states:
            # Find closest state
            closest_state = self._find_closest_state(current_state)
            current_state = closest_state
        
        # Get optimal action
        optimal_action = self.mdp.get_optimal_action(current_state)
        
        if optimal_action is None:
            # Fallback to safe default
            optimal_action = self.mdp.actions.get("no_action")
        
        # Calculate expected utility
        expected_utility = self.mdp.value_function.get(current_state.state_id, 0.0)
        
        return optimal_action, expected_utility
    
    def _find_closest_state(self, target_state: HealthState) -> HealthState:
        """Find closest state in MDP to target state."""
        target_vector = target_state.to_vector()
        
        min_distance = float('inf')
        closest_state = None
        
        for state in self.mdp.states.values():
            state_vector = state.to_vector()
            distance = np.linalg.norm(target_vector - state_vector)
            
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        
        return closest_state or list(self.mdp.states.values())[0]
    
    def simulate_policy_performance(
        self, 
        initial_states: List[HealthState],
        simulation_steps: int = 20
    ) -> Dict[str, Any]:
        """Simulate policy performance across multiple initial states."""
        simulation_results = {
            "total_simulations": len(initial_states),
            "average_total_reward": 0.0,
            "success_rate": 0.0,
            "trajectories": []
        }
        
        total_rewards = []
        successful_outcomes = 0
        
        for initial_state in initial_states:
            trajectory = self.mdp.simulate_trajectory(
                initial_state, 
                num_steps=simulation_steps,
                use_optimal_policy=True
            )
            
            # Calculate total reward
            total_reward = sum(reward for _, _, reward in trajectory)
            total_rewards.append(total_reward)
            
            # Check if outcome was successful (improved health)
            if trajectory:
                final_state = trajectory[-1][0]
                if final_state.overall_wellness > initial_state.overall_wellness:
                    successful_outcomes += 1
            
            simulation_results["trajectories"].append({
                "initial_state": initial_state.state_id,
                "total_reward": total_reward,
                "trajectory_length": len(trajectory),
                "final_wellness": trajectory[-1][0].overall_wellness if trajectory else 0.0
            })
        
        simulation_results["average_total_reward"] = np.mean(total_rewards)
        simulation_results["success_rate"] = successful_outcomes / len(initial_states)
        simulation_results["reward_std"] = np.std(total_rewards)
        
        return simulation_results 
"""
Utility Optimization for Health Goals

This module implements utility functions and multi-objective optimization
for health decision making, allowing the system to balance competing
objectives and preferences.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import scipy.optimize
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class UtilityType(Enum):
    """Types of utility functions for health optimization."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential" 
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    PIECEWISE = "piecewise"
    CUSTOM = "custom"


@dataclass
class HealthGoal:
    """Represents a health optimization goal with utility preferences."""
    goal_id: str
    goal_name: str
    target_metric: str  # What health metric to optimize
    target_value: float  # Desired value
    priority: float = 1.0  # Goal priority (higher = more important)
    tolerance: float = 0.1  # Acceptable deviation from target
    time_horizon: int = 30  # Days to achieve goal
    
    # Utility function parameters
    utility_type: UtilityType = UtilityType.LINEAR
    utility_params: Dict[str, float] = field(default_factory=dict)
    
    # Constraints
    min_acceptable: Optional[float] = None
    max_acceptable: Optional[float] = None
    
    # Context
    is_critical: bool = False
    dependencies: List[str] = field(default_factory=list)  # Other goals this depends on


class UtilityFunction(ABC):
    """Abstract base class for utility functions."""
    
    @abstractmethod
    def calculate_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        """Calculate utility score for current vs target value."""
        pass
    
    @abstractmethod
    def calculate_marginal_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        """Calculate marginal utility (derivative)."""
        pass


class LinearUtility(UtilityFunction):
    """Linear utility function - utility increases linearly with proximity to target."""
    
    def calculate_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        max_deviation = kwargs.get('max_deviation', 1.0)
        deviation = abs(current_value - target_value)
        return max(0.0, 1.0 - (deviation / max_deviation))
    
    def calculate_marginal_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        max_deviation = kwargs.get('max_deviation', 1.0)
        if current_value > target_value:
            return -1.0 / max_deviation
        else:
            return 1.0 / max_deviation


class ExponentialUtility(UtilityFunction):
    """Exponential utility function - rapid increase near target."""
    
    def calculate_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        decay_rate = kwargs.get('decay_rate', 2.0)
        deviation = abs(current_value - target_value)
        max_deviation = kwargs.get('max_deviation', 1.0)
        normalized_deviation = deviation / max_deviation
        return np.exp(-decay_rate * normalized_deviation)
    
    def calculate_marginal_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        decay_rate = kwargs.get('decay_rate', 2.0)
        max_deviation = kwargs.get('max_deviation', 1.0)
        deviation = abs(current_value - target_value)
        normalized_deviation = deviation / max_deviation
        
        utility = np.exp(-decay_rate * normalized_deviation)
        sign = 1 if current_value < target_value else -1
        return sign * decay_rate * utility / max_deviation


class LogarithmicUtility(UtilityFunction):
    """Logarithmic utility function - diminishing returns."""
    
    def calculate_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        if current_value <= 0:
            return 0.0
        
        # Logarithmic utility with diminishing returns
        scale = kwargs.get('scale', 1.0)
        return scale * np.log(1 + current_value / max(target_value, 0.1))
    
    def calculate_marginal_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        if current_value <= 0:
            return float('inf')
        
        scale = kwargs.get('scale', 1.0)
        return scale / (current_value + max(target_value, 0.1))


class SigmoidUtility(UtilityFunction):
    """Sigmoid utility function - S-shaped curve with smooth transitions."""
    
    def calculate_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        steepness = kwargs.get('steepness', 1.0)
        midpoint = kwargs.get('midpoint', target_value)
        
        # Sigmoid centered at midpoint
        x = steepness * (current_value - midpoint)
        return 1.0 / (1.0 + np.exp(-x))
    
    def calculate_marginal_utility(self, current_value: float, target_value: float, **kwargs) -> float:
        steepness = kwargs.get('steepness', 1.0)
        midpoint = kwargs.get('midpoint', target_value)
        
        x = steepness * (current_value - midpoint)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return steepness * sigmoid * (1.0 - sigmoid)


class UtilityOptimizer:
    """
    Multi-objective utility optimizer for health goals.
    
    This class manages multiple competing health objectives and finds
    optimal solutions that balance different utility functions.
    """
    
    def __init__(
        self,
        optimization_method: str = "pareto",
        aggregation_method: str = "weighted_sum"
    ):
        self.optimization_method = optimization_method
        self.aggregation_method = aggregation_method
        
        # Goals and utilities
        self.goals: Dict[str, HealthGoal] = {}
        self.utility_functions: Dict[UtilityType, UtilityFunction] = {
            UtilityType.LINEAR: LinearUtility(),
            UtilityType.EXPONENTIAL: ExponentialUtility(),
            UtilityType.LOGARITHMIC: LogarithmicUtility(),
            UtilityType.SIGMOID: SigmoidUtility()
        }
        
        # Optimization state
        self.current_solution: Optional[Dict[str, float]] = None
        self.pareto_frontier: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"UtilityOptimizer initialized with {optimization_method} method")
    
    def add_goal(self, goal: HealthGoal) -> None:
        """Add a health goal to the optimization problem."""
        self.goals[goal.goal_id] = goal
        logger.info(f"Added goal: {goal.goal_name}")
    
    def remove_goal(self, goal_id: str) -> None:
        """Remove a goal from optimization."""
        if goal_id in self.goals:
            del self.goals[goal_id]
            logger.info(f"Removed goal: {goal_id}")
    
    def update_goal_priority(self, goal_id: str, new_priority: float) -> None:
        """Update the priority of a goal."""
        if goal_id in self.goals:
            self.goals[goal_id].priority = new_priority
            logger.info(f"Updated priority for {goal_id}: {new_priority}")
    
    def calculate_individual_utility(
        self, 
        goal_id: str, 
        current_values: Dict[str, float]
    ) -> float:
        """Calculate utility for a single goal."""
        if goal_id not in self.goals:
            return 0.0
        
        goal = self.goals[goal_id]
        current_value = current_values.get(goal.target_metric, 0.0)
        
        # Get utility function
        utility_func = self.utility_functions[goal.utility_type]
        
        # Calculate utility with goal-specific parameters
        utility_params = goal.utility_params.copy()
        utility_params.update({
            'max_deviation': goal.tolerance,
            'target_value': goal.target_value
        })
        
        utility = utility_func.calculate_utility(
            current_value, 
            goal.target_value, 
            **utility_params
        )
        
        return utility
    
    def calculate_aggregate_utility(
        self, 
        current_values: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate aggregate utility across all goals."""
        if not self.goals:
            return 0.0
        
        individual_utilities = {}
        total_priority = 0.0
        
        for goal_id, goal in self.goals.items():
            utility = self.calculate_individual_utility(goal_id, current_values)
            individual_utilities[goal_id] = utility
            total_priority += goal.priority
        
        # Aggregate based on method
        if self.aggregation_method == "weighted_sum":
            aggregate = 0.0
            for goal_id, utility in individual_utilities.items():
                weight = weights.get(goal_id, self.goals[goal_id].priority) if weights else self.goals[goal_id].priority
                normalized_weight = weight / total_priority if total_priority > 0 else 1.0 / len(self.goals)
                aggregate += normalized_weight * utility
            
        elif self.aggregation_method == "geometric_mean":
            utilities = list(individual_utilities.values())
            aggregate = np.power(np.prod(utilities), 1.0 / len(utilities))
            
        elif self.aggregation_method == "minimum":
            # Egalitarian approach - focus on worst-performing goal
            aggregate = min(individual_utilities.values())
            
        elif self.aggregation_method == "lexicographic":
            # Prioritize goals in order of priority
            sorted_goals = sorted(self.goals.items(), key=lambda x: x[1].priority, reverse=True)
            aggregate = individual_utilities[sorted_goals[0][0]]  # Use highest priority goal
            
        else:
            # Default to weighted sum
            aggregate = sum(individual_utilities.values()) / len(individual_utilities)
        
        return aggregate
    
    def optimize_single_objective(
        self, 
        goal_id: str,
        current_values: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Optimize for a single goal."""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        
        # Define objective function
        def objective(x):
            # x represents the decision variables (health metric adjustments)
            adjusted_values = current_values.copy()
            adjusted_values[goal.target_metric] = x[0]
            return -self.calculate_individual_utility(goal_id, adjusted_values)  # Minimize negative utility
        
        # Set up constraints
        bounds = []
        if goal.min_acceptable is not None and goal.max_acceptable is not None:
            bounds.append((goal.min_acceptable, goal.max_acceptable))
        else:
            # Default bounds
            current_val = current_values.get(goal.target_metric, goal.target_value)
            bounds.append((max(0, current_val - goal.tolerance * 3), current_val + goal.tolerance * 3))
        
        # Initial guess
        x0 = [current_values.get(goal.target_metric, goal.target_value)]
        
        # Optimize
        result = scipy.optimize.minimize(
            objective,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimal_value = result.x[0] if result.success else x0[0]
        max_utility = -result.fun if result.success else objective(x0)
        
        return {
            "goal_id": goal_id,
            "optimal_value": optimal_value,
            "maximum_utility": max_utility,
            "success": result.success,
            "message": result.message if hasattr(result, 'message') else 'Unknown',
            "optimization_details": result
        }
    
    def optimize_multi_objective(
        self, 
        current_values: Dict[str, float],
        method: str = "weighted_sum"
    ) -> Dict[str, Any]:
        """Optimize for multiple objectives simultaneously."""
        if not self.goals:
            return {"error": "No goals defined"}
        
        if method == "weighted_sum":
            return self._optimize_weighted_sum(current_values)
        elif method == "pareto":
            return self._optimize_pareto(current_values)
        elif method == "epsilon_constraint":
            return self._optimize_epsilon_constraint(current_values)
        else:
            raise ValueError(f"Unknown multi-objective optimization method: {method}")
    
    def _optimize_weighted_sum(self, current_values: Dict[str, float]) -> Dict[str, Any]:
        """Optimize using weighted sum approach."""
        # Decision variables: adjustments to each health metric
        metrics = list(set(goal.target_metric for goal in self.goals.values()))
        
        def objective(x):
            # x represents adjustments to each metric
            adjusted_values = current_values.copy()
            for i, metric in enumerate(metrics):
                adjusted_values[metric] = x[i]
            
            return -self.calculate_aggregate_utility(adjusted_values)  # Minimize negative utility
        
        # Set up bounds for each metric
        bounds = []
        x0 = []
        
        for metric in metrics:
            current_val = current_values.get(metric, 0.5)
            x0.append(current_val)
            
            # Find bounds based on goals for this metric
            metric_goals = [g for g in self.goals.values() if g.target_metric == metric]
            if metric_goals:
                min_bounds = [g.min_acceptable for g in metric_goals if g.min_acceptable is not None]
                max_bounds = [g.max_acceptable for g in metric_goals if g.max_acceptable is not None]
                
                lower = min(min_bounds) if min_bounds else max(0, current_val - 0.5)
                upper = max(max_bounds) if max_bounds else min(1, current_val + 0.5)
            else:
                lower, upper = max(0, current_val - 0.5), min(1, current_val + 0.5)
            
            bounds.append((lower, upper))
        
        # Optimize
        result = scipy.optimize.minimize(
            objective,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Extract optimal values
        optimal_values = {}
        if result.success:
            for i, metric in enumerate(metrics):
                optimal_values[metric] = result.x[i]
        else:
            optimal_values = {metric: x0[i] for i, metric in enumerate(metrics)}
        
        # Calculate individual utilities at optimal point
        individual_utilities = {}
        for goal_id in self.goals:
            individual_utilities[goal_id] = self.calculate_individual_utility(goal_id, optimal_values)
        
        aggregate_utility = -result.fun if result.success else -objective(x0)
        
        return {
            "method": "weighted_sum",
            "optimal_values": optimal_values,
            "aggregate_utility": aggregate_utility,
            "individual_utilities": individual_utilities,
            "success": result.success,
            "optimization_details": result
        }
    
    def _optimize_pareto(self, current_values: Dict[str, float]) -> Dict[str, Any]:
        """Find Pareto-optimal solutions."""
        pareto_solutions = []
        
        # Generate multiple weight combinations for Pareto frontier
        num_goals = len(self.goals)
        if num_goals <= 1:
            return self._optimize_weighted_sum(current_values)
        
        # Create weight vectors for Pareto frontier exploration
        num_points = 20  # Number of Pareto points to find
        weight_combinations = self._generate_weight_combinations(num_goals, num_points)
        
        for weights in weight_combinations:
            # Temporarily adjust goal priorities
            original_priorities = {}
            goal_ids = list(self.goals.keys())
            
            for i, goal_id in enumerate(goal_ids):
                original_priorities[goal_id] = self.goals[goal_id].priority
                self.goals[goal_id].priority = weights[i]
            
            # Optimize with these weights
            solution = self._optimize_weighted_sum(current_values)
            
            if solution.get("success", False):
                pareto_solutions.append({
                    "weights": weights.copy(),
                    "solution": solution
                })
            
            # Restore original priorities
            for goal_id, priority in original_priorities.items():
                self.goals[goal_id].priority = priority
        
        # Filter for truly Pareto-optimal solutions
        pareto_optimal = self._filter_pareto_optimal(pareto_solutions)
        
        # Store Pareto frontier
        self.pareto_frontier = pareto_optimal
        
        # Return best solution (highest aggregate utility)
        if pareto_optimal:
            best_solution = max(pareto_optimal, key=lambda x: x["solution"]["aggregate_utility"])
            best_solution["method"] = "pareto"
            best_solution["pareto_frontier_size"] = len(pareto_optimal)
            return best_solution["solution"]
        
        return {"error": "No Pareto-optimal solutions found"}
    
    def _generate_weight_combinations(self, num_goals: int, num_points: int) -> List[List[float]]:
        """Generate diverse weight combinations for multi-objective optimization."""
        if num_goals == 2:
            # For 2 goals, use linear spacing
            alphas = np.linspace(0.1, 0.9, num_points)
            return [[alpha, 1.0 - alpha] for alpha in alphas]
        
        # For more goals, use random sampling with simplex constraint
        combinations = []
        for _ in range(num_points):
            # Random weights that sum to 1
            weights = np.random.exponential(1, num_goals)
            weights = weights / np.sum(weights)
            combinations.append(weights.tolist())
        
        return combinations
    
    def _filter_pareto_optimal(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter solutions to keep only Pareto-optimal ones."""
        if not solutions:
            return []
        
        pareto_optimal = []
        
        for i, solution_i in enumerate(solutions):
            is_dominated = False
            utilities_i = solution_i["solution"]["individual_utilities"]
            
            for j, solution_j in enumerate(solutions):
                if i == j:
                    continue
                
                utilities_j = solution_j["solution"]["individual_utilities"]
                
                # Check if solution_j dominates solution_i
                dominates = True
                strictly_better = False
                
                for goal_id in utilities_i:
                    if goal_id in utilities_j:
                        if utilities_j[goal_id] < utilities_i[goal_id]:
                            dominates = False
                            break
                        elif utilities_j[goal_id] > utilities_i[goal_id]:
                            strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(solution_i)
        
        return pareto_optimal
    
    def recommend_goal_adjustments(
        self, 
        current_values: Dict[str, float],
        optimization_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific recommendations for achieving optimal values."""
        recommendations = []
        
        if "optimal_values" not in optimization_result:
            return recommendations
        
        optimal_values = optimization_result["optimal_values"]
        
        for metric, optimal_value in optimal_values.items():
            current_value = current_values.get(metric, 0.0)
            
            if abs(optimal_value - current_value) > 0.01:  # Threshold for significant change
                change_direction = "increase" if optimal_value > current_value else "decrease"
                change_magnitude = abs(optimal_value - current_value)
                
                # Find goals related to this metric
                related_goals = [g for g in self.goals.values() if g.target_metric == metric]
                
                recommendation = {
                    "metric": metric,
                    "current_value": current_value,
                    "optimal_value": optimal_value,
                    "change_direction": change_direction,
                    "change_magnitude": change_magnitude,
                    "priority": "high" if change_magnitude > 0.2 else "medium" if change_magnitude > 0.1 else "low",
                    "related_goals": [g.goal_name for g in related_goals],
                    "time_horizon": min(g.time_horizon for g in related_goals) if related_goals else 30
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority and magnitude
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda x: (priority_order[x["priority"]], x["change_magnitude"]), 
            reverse=True
        )
        
        return recommendations
    
    def get_utility_landscape(
        self, 
        current_values: Dict[str, float],
        metric: str,
        value_range: Tuple[float, float],
        num_points: int = 100
    ) -> Dict[str, Any]:
        """Get utility landscape for a specific metric."""
        values = np.linspace(value_range[0], value_range[1], num_points)
        utilities = []
        
        for value in values:
            test_values = current_values.copy()
            test_values[metric] = value
            utility = self.calculate_aggregate_utility(test_values)
            utilities.append(utility)
        
        # Find optimal point
        max_idx = np.argmax(utilities)
        optimal_value = values[max_idx]
        max_utility = utilities[max_idx]
        
        return {
            "metric": metric,
            "values": values.tolist(),
            "utilities": utilities,
            "optimal_value": optimal_value,
            "max_utility": max_utility,
            "current_value": current_values.get(metric, 0.0),
            "current_utility": self.calculate_aggregate_utility(current_values)
        }
    
    def analyze_goal_conflicts(self) -> List[Dict[str, Any]]:
        """Analyze conflicts between goals."""
        conflicts = []
        
        goal_list = list(self.goals.values())
        for i, goal1 in enumerate(goal_list):
            for j, goal2 in enumerate(goal_list[i+1:], i+1):
                # Check if goals target the same metric with different values
                if goal1.target_metric == goal2.target_metric:
                    if abs(goal1.target_value - goal2.target_value) > max(goal1.tolerance, goal2.tolerance):
                        conflict = {
                            "goal1": goal1.goal_id,
                            "goal2": goal2.goal_id,
                            "conflict_type": "same_metric_different_targets",
                            "metric": goal1.target_metric,
                            "target1": goal1.target_value,
                            "target2": goal2.target_value,
                            "severity": abs(goal1.target_value - goal2.target_value) / max(goal1.tolerance, goal2.tolerance)
                        }
                        conflicts.append(conflict)
                
                # Check for dependency conflicts
                if goal2.goal_id in goal1.dependencies and goal1.goal_id in goal2.dependencies:
                    conflict = {
                        "goal1": goal1.goal_id,
                        "goal2": goal2.goal_id,
                        "conflict_type": "circular_dependency",
                        "severity": 1.0
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimization state."""
        return {
            "total_goals": len(self.goals),
            "optimization_method": self.optimization_method,
            "aggregation_method": self.aggregation_method,
            "goals_by_priority": {
                goal_id: goal.priority 
                for goal_id, goal in sorted(self.goals.items(), key=lambda x: x[1].priority, reverse=True)
            },
            "pareto_frontier_size": len(self.pareto_frontier),
            "goal_conflicts": len(self.analyze_goal_conflicts()),
            "has_current_solution": self.current_solution is not None
        } 
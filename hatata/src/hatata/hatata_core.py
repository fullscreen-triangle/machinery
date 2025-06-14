"""
Hatata Core: Stochastic Health Decision System

This is the main orchestration engine that combines Markov Decision Processes,
utility optimization, and stochastic modeling to provide evidence-based
health decision making under uncertainty.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .mdp_engine import MDPEngine, HealthState, HealthAction
from .utility_optimizer import UtilityOptimizer, HealthGoal, UtilityType

logger = logging.getLogger(__name__)


class DecisionConfidence(Enum):
    """Confidence levels for stochastic decisions."""
    VERY_LOW = "very_low"    # < 0.3
    LOW = "low"              # 0.3 - 0.5
    MODERATE = "moderate"    # 0.5 - 0.7
    HIGH = "high"            # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class StochasticEvidence:
    """
    Evidence from stochastic analysis for health decisions.
    """
    evidence_id: str
    source: str  # "mdp", "utility", "stochastic_model", "integration"
    
    # Evidence content
    decision_recommendation: str
    confidence_score: float
    uncertainty_bounds: Tuple[float, float]
    
    # Supporting data
    expected_value: float
    risk_assessment: Dict[str, float]
    time_horizon: int
    
    # Stochastic properties
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    monte_carlo_runs: int = 1000
    convergence_achieved: bool = True
    
    # Meta information
    timestamp: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def confidence_level(self) -> DecisionConfidence:
        """Get qualitative confidence level."""
        if self.confidence_score < 0.3:
            return DecisionConfidence.VERY_LOW
        elif self.confidence_score < 0.5:
            return DecisionConfidence.LOW
        elif self.confidence_score < 0.7:
            return DecisionConfidence.MODERATE
        elif self.confidence_score < 0.9:
            return DecisionConfidence.HIGH
        else:
            return DecisionConfidence.VERY_HIGH


@dataclass
class HatataDecision:
    """
    A comprehensive health decision with stochastic backing.
    """
    decision_id: str
    decision_type: str
    recommended_action: str
    
    # Decision quality metrics
    expected_utility: float
    confidence_score: float
    risk_score: float
    
    # Stochastic support
    evidence_sources: List[StochasticEvidence]
    monte_carlo_results: Dict[str, Any]
    
    # Implementation details
    implementation_steps: List[Dict[str, Any]]
    resource_requirements: Dict[str, float]
    time_to_effect: float
    
    # Uncertainty quantification
    uncertainty_analysis: Dict[str, Any]
    robustness_score: float
    
    # Context
    current_state: HealthState
    target_goals: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class HatataCore:
    """
    Core orchestration engine for stochastic health decision making.
    
    This class integrates:
    1. Markov Decision Processes for state transition modeling
    2. Utility optimization for multi-objective goal handling
    3. Stochastic processes for uncertainty quantification
    4. Evidence integration from mzekezeke and diggiden
    """
    
    def __init__(
        self,
        mdp_config: Optional[Dict[str, Any]] = None,
        utility_config: Optional[Dict[str, Any]] = None,
        stochastic_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize core components
        self.mdp_engine = MDPEngine()
        self.utility_optimizer = UtilityOptimizer(
            optimization_method=utility_config.get("method", "pareto") if utility_config else "pareto",
            aggregation_method=utility_config.get("aggregation", "weighted_sum") if utility_config else "weighted_sum"
        )
        
        # Stochastic configuration
        self.monte_carlo_runs = stochastic_config.get("monte_carlo_runs", 1000) if stochastic_config else 1000
        self.confidence_threshold = stochastic_config.get("confidence_threshold", 0.7) if stochastic_config else 0.7
        self.uncertainty_tolerance = stochastic_config.get("uncertainty_tolerance", 0.2) if stochastic_config else 0.2
        
        # System state
        self.current_health_state: Optional[HealthState] = None
        self.decision_history: List[HatataDecision] = []
        self.evidence_cache: Dict[str, StochasticEvidence] = {}
        
        # Integration state
        self.mzekezeke_predictions: Dict[str, Any] = {}
        self.diggiden_challenges: Dict[str, Any] = {}
        
        # Learning and adaptation
        self.decision_outcomes: List[Dict[str, Any]] = []
        self.adaptation_rate = 0.1
        
        logger.info("HatataCore initialized with stochastic decision capabilities")
    
    def set_current_state(self, health_state: HealthState) -> None:
        """Set the current health state for decision making."""
        self.current_health_state = health_state
        logger.info(f"Current health state set: {health_state.state_id}")
    
    def add_health_goal(
        self,
        goal_name: str,
        target_metric: str,
        target_value: float,
        priority: float = 1.0,
        utility_type: UtilityType = UtilityType.LINEAR,
        **kwargs
    ) -> str:
        """Add a health goal for optimization."""
        goal_id = f"goal_{len(self.utility_optimizer.goals)}_{goal_name.lower().replace(' ', '_')}"
        
        goal = HealthGoal(
            goal_id=goal_id,
            goal_name=goal_name,
            target_metric=target_metric,
            target_value=target_value,
            priority=priority,
            utility_type=utility_type,
            **kwargs
        )
        
        self.utility_optimizer.add_goal(goal)
        logger.info(f"Added health goal: {goal_name}")
        
        return goal_id
    
    def analyze_decision_problem(
        self,
        current_metrics: Dict[str, float],
        time_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze the current decision problem using stochastic methods.
        
        Returns comprehensive analysis including:
        - MDP policy recommendations
        - Utility optimization results
        - Stochastic uncertainty analysis
        - Integrated evidence assessment
        """
        analysis_start = datetime.now()
        
        # 1. MDP Analysis
        mdp_analysis = self._analyze_mdp_decisions(current_metrics, time_horizon)
        
        # 2. Utility Optimization
        utility_analysis = self._analyze_utility_optimization(current_metrics)
        
        # 3. Stochastic Uncertainty Analysis
        stochastic_analysis = self._analyze_stochastic_uncertainty(current_metrics, time_horizon)
        
        # 4. Evidence Integration
        integrated_evidence = self._integrate_evidence_sources(
            mdp_analysis, utility_analysis, stochastic_analysis
        )
        
        # 5. Decision Synthesis
        final_recommendation = self._synthesize_decision(integrated_evidence)
        
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        return {
            "analysis_timestamp": analysis_start.isoformat(),
            "analysis_duration": analysis_time,
            "mdp_analysis": mdp_analysis,
            "utility_analysis": utility_analysis,
            "stochastic_analysis": stochastic_analysis,
            "integrated_evidence": integrated_evidence,
            "final_recommendation": final_recommendation,
            "confidence_metrics": self._calculate_confidence_metrics(integrated_evidence)
        }
    
    def _analyze_mdp_decisions(
        self,
        current_metrics: Dict[str, float],
        time_horizon: int
    ) -> Dict[str, Any]:
        """Analyze decisions using MDP framework."""
        if not self.current_health_state:
            # Create state from metrics
            state_vector = np.array(list(current_metrics.values()) + [0.8] * (17 - len(current_metrics)))
            self.current_health_state = HealthState.from_vector(state_vector)
        
        # Optimize MDP policy
        policy_results = self.mdp_engine.optimize_policy(method="value_iteration")
        
        # Get action recommendation
        action, expected_utility = self.mdp_engine.recommend_action(self.current_health_state)
        
        # Simulate future trajectories
        simulation_results = self.mdp_engine.simulate_policy_performance(
            [self.current_health_state], 
            simulation_steps=time_horizon
        )
        
        # Monte Carlo simulation for robustness
        monte_carlo_trajectories = []
        for _ in range(min(100, self.monte_carlo_runs // 10)):  # Reduced for MDP
            trajectory = self.mdp_engine.mdp.simulate_trajectory(
                self.current_health_state,
                num_steps=time_horizon,
                use_optimal_policy=True
            )
            monte_carlo_trajectories.append(trajectory)
        
        return {
            "optimal_action": action.action_id if action else None,
            "expected_utility": expected_utility,
            "policy_optimization": policy_results,
            "simulation_results": simulation_results,
            "monte_carlo_trajectories": len(monte_carlo_trajectories),
            "robustness_metrics": self._calculate_mdp_robustness(monte_carlo_trajectories)
        }
    
    def _analyze_utility_optimization(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze decisions using utility optimization."""
        if not self.utility_optimizer.goals:
            return {"error": "No utility goals defined"}
        
        # Single objective optimizations
        single_objective_results = {}
        for goal_id in self.utility_optimizer.goals:
            result = self.utility_optimizer.optimize_single_objective(goal_id, current_metrics)
            single_objective_results[goal_id] = result
        
        # Multi-objective optimization
        multi_objective_result = self.utility_optimizer.optimize_multi_objective(
            current_metrics, method="pareto"
        )
        
        # Goal conflict analysis
        conflicts = self.utility_optimizer.analyze_goal_conflicts()
        
        # Generate recommendations
        recommendations = []
        if "optimal_values" in multi_objective_result:
            recommendations = self.utility_optimizer.recommend_goal_adjustments(
                current_metrics, multi_objective_result
            )
        
        return {
            "single_objective_results": single_objective_results,
            "multi_objective_result": multi_objective_result,
            "goal_conflicts": conflicts,
            "recommendations": recommendations,
            "pareto_frontier_size": len(self.utility_optimizer.pareto_frontier)
        }
    
    def _analyze_stochastic_uncertainty(
        self,
        current_metrics: Dict[str, float],
        time_horizon: int
    ) -> Dict[str, Any]:
        """Analyze uncertainty using stochastic methods."""
        # Monte Carlo simulation of metric evolution
        metric_trajectories = self._simulate_metric_trajectories(current_metrics, time_horizon)
        
        # Uncertainty quantification
        uncertainty_metrics = self._quantify_uncertainty(metric_trajectories)
        
        # Risk assessment
        risk_assessment = self._assess_risks(metric_trajectories)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(metric_trajectories)
        
        return {
            "monte_carlo_runs": self.monte_carlo_runs,
            "trajectory_statistics": self._calculate_trajectory_statistics(metric_trajectories),
            "uncertainty_metrics": uncertainty_metrics,
            "risk_assessment": risk_assessment,
            "confidence_intervals": confidence_intervals,
            "convergence_analysis": self._analyze_monte_carlo_convergence(metric_trajectories)
        }
    
    def _simulate_metric_trajectories(
        self,
        current_metrics: Dict[str, float],
        time_horizon: int
    ) -> Dict[str, List[List[float]]]:
        """Simulate future trajectories of health metrics using stochastic processes."""
        trajectories = {}
        
        for metric, current_value in current_metrics.items():
            metric_trajectories = []
            
            for _ in range(self.monte_carlo_runs):
                trajectory = [current_value]
                value = current_value
                
                for t in range(time_horizon):
                    # Stochastic evolution with mean reversion and noise
                    drift = 0.01 * (0.7 - value)  # Mean reversion to 0.7
                    volatility = 0.05  # Base volatility
                    noise = np.random.normal(0, volatility)
                    
                    # Evolution equation
                    value = max(0.0, min(1.0, value + drift + noise))
                    trajectory.append(value)
                
                metric_trajectories.append(trajectory)
            
            trajectories[metric] = metric_trajectories
        
        return trajectories
    
    def _quantify_uncertainty(self, trajectories: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Quantify uncertainty in trajectory projections."""
        uncertainty_metrics = {}
        
        for metric, metric_trajectories in trajectories.items():
            # Convert to array for easier computation
            traj_array = np.array(metric_trajectories)  # Shape: (runs, time_steps)
            
            # Calculate uncertainty metrics
            mean_trajectory = np.mean(traj_array, axis=0)
            std_trajectory = np.std(traj_array, axis=0)
            
            # Uncertainty measures
            total_variance = np.var(traj_array[:, -1])  # Final time variance
            path_variance = np.mean(np.var(traj_array, axis=1))  # Average path-wise variance
            
            uncertainty_metrics[metric] = {
                "total_variance": total_variance,
                "path_variance": path_variance,
                "final_std": std_trajectory[-1],
                "mean_final_value": mean_trajectory[-1],
                "uncertainty_coefficient": std_trajectory[-1] / max(mean_trajectory[-1], 0.1)
            }
        
        return uncertainty_metrics
    
    def _assess_risks(self, trajectories: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Assess risks based on trajectory analysis."""
        risk_assessment = {}
        
        for metric, metric_trajectories in trajectories.items():
            traj_array = np.array(metric_trajectories)
            final_values = traj_array[:, -1]
            
            # Risk metrics
            prob_below_50 = np.mean(final_values < 0.5)
            prob_below_30 = np.mean(final_values < 0.3)
            prob_above_80 = np.mean(final_values > 0.8)
            
            # Value at Risk (VaR) - 5th percentile
            var_5 = np.percentile(final_values, 5)
            var_10 = np.percentile(final_values, 10)
            
            # Expected Shortfall (CVaR)
            cvar_5 = np.mean(final_values[final_values <= var_5])
            
            risk_assessment[metric] = {
                "probability_decline": prob_below_50,
                "probability_severe_decline": prob_below_30,
                "probability_improvement": prob_above_80,
                "value_at_risk_5": var_5,
                "value_at_risk_10": var_10,
                "conditional_value_at_risk": cvar_5,
                "risk_score": (prob_below_30 * 0.5 + prob_below_50 * 0.3 + (1.0 - var_10) * 0.2)
            }
        
        return risk_assessment
    
    def _calculate_confidence_intervals(self, trajectories: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Calculate confidence intervals for projections."""
        confidence_intervals = {}
        
        for metric, metric_trajectories in trajectories.items():
            traj_array = np.array(metric_trajectories)
            
            # Calculate percentiles over time
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            interval_data = {}
            
            for p in percentiles:
                interval_data[f"p{p}"] = np.percentile(traj_array, p, axis=0).tolist()
            
            confidence_intervals[metric] = interval_data
        
        return confidence_intervals
    
    def _integrate_evidence_sources(
        self,
        mdp_analysis: Dict[str, Any],
        utility_analysis: Dict[str, Any],
        stochastic_analysis: Dict[str, Any]
    ) -> List[StochasticEvidence]:
        """Integrate evidence from different analytical sources."""
        evidence_list = []
        
        # MDP Evidence
        if mdp_analysis.get("optimal_action"):
            mdp_evidence = StochasticEvidence(
                evidence_id="mdp_policy",
                source="mdp",
                decision_recommendation=mdp_analysis["optimal_action"],
                confidence_score=mdp_analysis.get("robustness_metrics", {}).get("confidence", 0.7),
                uncertainty_bounds=(0.6, 0.9),  # Simplified
                expected_value=mdp_analysis["expected_utility"],
                risk_assessment=mdp_analysis.get("robustness_metrics", {}),
                time_horizon=30,
                monte_carlo_runs=mdp_analysis.get("monte_carlo_trajectories", 0)
            )
            evidence_list.append(mdp_evidence)
        
        # Utility Evidence
        if "multi_objective_result" in utility_analysis and utility_analysis["multi_objective_result"].get("success"):
            utility_result = utility_analysis["multi_objective_result"]
            
            utility_evidence = StochasticEvidence(
                evidence_id="utility_optimization",
                source="utility",
                decision_recommendation=f"optimize_metrics_{len(utility_result.get('optimal_values', {}))}",
                confidence_score=min(0.9, utility_result.get("aggregate_utility", 0.5) + 0.2),
                uncertainty_bounds=(0.5, 0.95),
                expected_value=utility_result.get("aggregate_utility", 0.0),
                risk_assessment={"conflicts": len(utility_analysis.get("goal_conflicts", []))},
                time_horizon=30
            )
            evidence_list.append(utility_evidence)
        
        # Stochastic Evidence
        stochastic_evidence = StochasticEvidence(
            evidence_id="stochastic_analysis",
            source="stochastic_model",
            decision_recommendation="uncertainty_informed_decision",
            confidence_score=self._calculate_stochastic_confidence(stochastic_analysis),
            uncertainty_bounds=self._calculate_stochastic_bounds(stochastic_analysis),
            expected_value=0.0,  # Will be calculated
            risk_assessment=stochastic_analysis.get("risk_assessment", {}),
            time_horizon=30,
            monte_carlo_runs=self.monte_carlo_runs,
            convergence_achieved=stochastic_analysis.get("convergence_analysis", {}).get("converged", True)
        )
        evidence_list.append(stochastic_evidence)
        
        return evidence_list
    
    def _calculate_stochastic_confidence(self, stochastic_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score from stochastic analysis."""
        uncertainty_metrics = stochastic_analysis.get("uncertainty_metrics", {})
        
        if not uncertainty_metrics:
            return 0.5
        
        # Average uncertainty coefficient across metrics
        uncertainty_coeffs = [
            metrics.get("uncertainty_coefficient", 0.5) 
            for metrics in uncertainty_metrics.values()
        ]
        
        avg_uncertainty = np.mean(uncertainty_coeffs)
        
        # Convert uncertainty to confidence (lower uncertainty = higher confidence)
        confidence = max(0.1, min(0.95, 1.0 - avg_uncertainty))
        
        return confidence
    
    def _calculate_stochastic_bounds(self, stochastic_analysis: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate uncertainty bounds from stochastic analysis."""
        risk_assessment = stochastic_analysis.get("risk_assessment", {})
        
        if not risk_assessment:
            return (0.3, 0.8)
        
        # Get risk scores across metrics
        risk_scores = [
            assessment.get("risk_score", 0.5)
            for assessment in risk_assessment.values()
        ]
        
        avg_risk = np.mean(risk_scores)
        
        # Convert to uncertainty bounds
        lower_bound = max(0.1, 0.7 - avg_risk)
        upper_bound = min(0.95, 0.8 + (1.0 - avg_risk) * 0.2)
        
        return (lower_bound, upper_bound)
    
    def _synthesize_decision(self, evidence_list: List[StochasticEvidence]) -> HatataDecision:
        """Synthesize final decision from all evidence sources."""
        if not evidence_list:
            return self._create_default_decision()
        
        # Weight evidence by confidence
        total_weight = sum(ev.confidence_score for ev in evidence_list)
        
        if total_weight == 0:
            return self._create_default_decision()
        
        # Calculate aggregate metrics
        weighted_expected_value = sum(
            ev.expected_value * ev.confidence_score for ev in evidence_list
        ) / total_weight
        
        aggregate_confidence = np.mean([ev.confidence_score for ev in evidence_list])
        
        # Risk aggregation
        risk_scores = []
        for ev in evidence_list:
            if isinstance(ev.risk_assessment, dict) and ev.risk_assessment:
                # Extract numerical risk scores
                numeric_risks = [
                    v for v in ev.risk_assessment.values() 
                    if isinstance(v, (int, float))
                ]
                if numeric_risks:
                    risk_scores.append(np.mean(numeric_risks))
        
        aggregate_risk = np.mean(risk_scores) if risk_scores else 0.3
        
        # Choose primary recommendation (highest confidence evidence)
        primary_evidence = max(evidence_list, key=lambda x: x.confidence_score)
        
        # Create decision
        decision = HatataDecision(
            decision_id=f"hatata_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type="stochastic_optimization",
            recommended_action=primary_evidence.decision_recommendation,
            expected_utility=weighted_expected_value,
            confidence_score=aggregate_confidence,
            risk_score=aggregate_risk,
            evidence_sources=evidence_list,
            monte_carlo_results={"runs": self.monte_carlo_runs, "convergence": True},
            implementation_steps=self._generate_implementation_steps(primary_evidence),
            resource_requirements={"time": 1.0, "effort": 0.5, "cost": 0.3},
            time_to_effect=primary_evidence.time_horizon,
            uncertainty_analysis=self._create_uncertainty_analysis(evidence_list),
            robustness_score=self._calculate_robustness_score(evidence_list),
            current_state=self.current_health_state,
            target_goals=list(self.utility_optimizer.goals.keys())
        )
        
        return decision
    
    def _create_default_decision(self) -> HatataDecision:
        """Create a default decision when no evidence is available."""
        return HatataDecision(
            decision_id="default_decision",
            decision_type="conservative",
            recommended_action="maintain_current_state",
            expected_utility=0.0,
            confidence_score=0.3,
            risk_score=0.5,
            evidence_sources=[],
            monte_carlo_results={},
            implementation_steps=[{"action": "monitor", "priority": "low"}],
            resource_requirements={"time": 0.1, "effort": 0.1, "cost": 0.0},
            time_to_effect=7,
            uncertainty_analysis={},
            robustness_score=0.3,
            current_state=self.current_health_state,
            target_goals=[]
        )
    
    def _generate_implementation_steps(self, evidence: StochasticEvidence) -> List[Dict[str, Any]]:
        """Generate implementation steps based on evidence."""
        steps = []
        
        if evidence.source == "mdp":
            steps.append({
                "action": "implement_mdp_policy",
                "description": f"Execute action: {evidence.decision_recommendation}",
                "priority": "high",
                "timeline": "immediate"
            })
        
        elif evidence.source == "utility":
            steps.append({
                "action": "optimize_utility_goals",
                "description": "Adjust health metrics toward utility-optimal values",
                "priority": "medium",
                "timeline": "gradual"
            })
        
        elif evidence.source == "stochastic_model":
            steps.append({
                "action": "uncertainty_management",
                "description": "Implement robust strategies under uncertainty",
                "priority": "medium",
                "timeline": "ongoing"
            })
        
        # Common monitoring step
        steps.append({
            "action": "monitor_outcomes",
            "description": "Track health metrics and decision effectiveness",
            "priority": "high",
            "timeline": "continuous"
        })
        
        return steps
    
    def _create_uncertainty_analysis(self, evidence_list: List[StochasticEvidence]) -> Dict[str, Any]:
        """Create comprehensive uncertainty analysis."""
        return {
            "evidence_agreement": self._calculate_evidence_agreement(evidence_list),
            "confidence_distribution": [ev.confidence_score for ev in evidence_list],
            "uncertainty_sources": [ev.source for ev in evidence_list],
            "total_evidence_pieces": len(evidence_list),
            "monte_carlo_validation": all(ev.convergence_achieved for ev in evidence_list)
        }
    
    def _calculate_evidence_agreement(self, evidence_list: List[StochasticEvidence]) -> float:
        """Calculate agreement between different evidence sources."""
        if len(evidence_list) < 2:
            return 1.0
        
        # Simple agreement based on confidence score similarity
        confidences = [ev.confidence_score for ev in evidence_list]
        std_confidence = np.std(confidences)
        
        # Lower standard deviation = higher agreement
        agreement = max(0.0, 1.0 - std_confidence * 2)
        
        return agreement
    
    def _calculate_robustness_score(self, evidence_list: List[StochasticEvidence]) -> float:
        """Calculate overall robustness of the decision."""
        if not evidence_list:
            return 0.3
        
        # Factors for robustness:
        # 1. Number of evidence sources
        # 2. Confidence levels
        # 3. Monte Carlo validation
        # 4. Evidence agreement
        
        source_diversity = len(set(ev.source for ev in evidence_list)) / 4.0  # Max 4 sources
        avg_confidence = np.mean([ev.confidence_score for ev in evidence_list])
        mc_validation = np.mean([float(ev.convergence_achieved) for ev in evidence_list])
        evidence_agreement = self._calculate_evidence_agreement(evidence_list)
        
        robustness = (
            source_diversity * 0.25 +
            avg_confidence * 0.35 +
            mc_validation * 0.20 +
            evidence_agreement * 0.20
        )
        
        return min(0.95, max(0.1, robustness))
    
    def _calculate_confidence_metrics(self, evidence_list: List[StochasticEvidence]) -> Dict[str, Any]:
        """Calculate comprehensive confidence metrics."""
        if not evidence_list:
            return {"overall_confidence": 0.3, "confidence_level": "low"}
        
        confidences = [ev.confidence_score for ev in evidence_list]
        
        return {
            "overall_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "confidence_level": DecisionConfidence.MODERATE.value,  # Simplified
            "evidence_count": len(evidence_list),
            "high_confidence_sources": sum(1 for c in confidences if c > 0.7)
        }
    
    def _calculate_mdp_robustness(self, trajectories: List[Any]) -> Dict[str, Any]:
        """Calculate robustness metrics for MDP analysis."""
        if not trajectories:
            return {"confidence": 0.5}
        
        # Extract rewards from trajectories
        trajectory_rewards = []
        for trajectory in trajectories:
            if trajectory:
                total_reward = sum(reward for _, _, reward in trajectory)
                trajectory_rewards.append(total_reward)
        
        if not trajectory_rewards:
            return {"confidence": 0.5}
        
        mean_reward = np.mean(trajectory_rewards)
        std_reward = np.std(trajectory_rewards)
        
        # Confidence based on reward consistency
        confidence = max(0.3, min(0.9, 0.7 - std_reward / max(abs(mean_reward), 1.0)))
        
        return {
            "confidence": confidence,
            "mean_reward": mean_reward,
            "reward_std": std_reward,
            "trajectory_count": len(trajectories)
        }
    
    def _calculate_trajectory_statistics(self, trajectories: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Calculate statistics for metric trajectories."""
        stats = {}
        
        for metric, metric_trajectories in trajectories.items():
            traj_array = np.array(metric_trajectories)
            
            stats[metric] = {
                "initial_mean": np.mean(traj_array[:, 0]),
                "final_mean": np.mean(traj_array[:, -1]),
                "mean_change": np.mean(traj_array[:, -1]) - np.mean(traj_array[:, 0]),
                "final_std": np.std(traj_array[:, -1]),
                "max_trajectory": np.max(traj_array[:, -1]),
                "min_trajectory": np.min(traj_array[:, -1])
            }
        
        return stats
    
    def _analyze_monte_carlo_convergence(self, trajectories: Dict[str, List[List[float]]]) -> Dict[str, Any]:
        """Analyze convergence of Monte Carlo simulation."""
        convergence_analysis = {}
        
        for metric, metric_trajectories in trajectories.items():
            traj_array = np.array(metric_trajectories)
            final_values = traj_array[:, -1]
            
            # Check convergence by comparing running averages
            running_means = np.cumsum(final_values) / np.arange(1, len(final_values) + 1)
            
            # Convergence test: last 10% should be stable
            stability_window = max(10, len(running_means) // 10)
            recent_means = running_means[-stability_window:]
            
            convergence_achieved = np.std(recent_means) < 0.01  # Arbitrary threshold
            
            convergence_analysis[metric] = {
                "converged": convergence_achieved,
                "final_mean": running_means[-1],
                "stability_std": np.std(recent_means),
                "convergence_window": stability_window
            }
        
        # Overall convergence
        all_converged = all(analysis["converged"] for analysis in convergence_analysis.values())
        
        return {
            "overall_converged": all_converged,
            "metric_convergence": convergence_analysis,
            "total_runs": self.monte_carlo_runs
        }
    
    def make_decision(
        self,
        current_metrics: Dict[str, float],
        time_horizon: int = 30,
        include_uncertainty: bool = True
    ) -> HatataDecision:
        """
        Make a comprehensive health decision using stochastic analysis.
        
        This is the main decision-making interface that orchestrates
        all analytical components.
        """
        logger.info("Making stochastic health decision")
        
        # Analyze the decision problem
        analysis_results = self.analyze_decision_problem(current_metrics, time_horizon)
        
        # Extract decision from analysis
        decision = analysis_results["final_recommendation"]
        
        # Store decision history
        self.decision_history.append(decision)
        
        # Update evidence cache
        for evidence in decision.evidence_sources:
            self.evidence_cache[evidence.evidence_id] = evidence
        
        logger.info(f"Decision made: {decision.recommended_action} (confidence: {decision.confidence_score:.2f})")
        
        return decision
    
    def get_decision_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for recent decisions."""
        if not self.decision_history:
            return {"error": "No decisions in history"}
        
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": np.mean([d.confidence_score for d in recent_decisions]),
            "average_risk": np.mean([d.risk_score for d in recent_decisions]),
            "average_robustness": np.mean([d.robustness_score for d in recent_decisions]),
            "decision_types": {
                d.decision_type: sum(1 for dec in recent_decisions if dec.decision_type == d.decision_type)
                for d in recent_decisions
            }
        }
    
    def export_decision_evidence(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Export comprehensive evidence for a specific decision."""
        decision = next((d for d in self.decision_history if d.decision_id == decision_id), None)
        
        if not decision:
            return None
        
        return {
            "decision": {
                "id": decision.decision_id,
                "type": decision.decision_type,
                "action": decision.recommended_action,
                "timestamp": decision.timestamp.isoformat()
            },
            "quality_metrics": {
                "expected_utility": decision.expected_utility,
                "confidence_score": decision.confidence_score,
                "risk_score": decision.risk_score,
                "robustness_score": decision.robustness_score
            },
            "evidence_sources": [
                {
                    "id": ev.evidence_id,
                    "source": ev.source,
                    "confidence": ev.confidence_score,
                    "recommendation": ev.decision_recommendation,
                    "monte_carlo_runs": ev.monte_carlo_runs
                }
                for ev in decision.evidence_sources
            ],
            "uncertainty_analysis": decision.uncertainty_analysis,
            "implementation_plan": decision.implementation_steps
        } 
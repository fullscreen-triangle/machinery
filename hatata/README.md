# hatata 🎲

*"Step by step"* - Stochastic health decision system using Markov Decision Processes and utility optimization

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Stochastic](https://img.shields.io/badge/stochastic-processes-blue?style=for-the-badge)
![MDP](https://img.shields.io/badge/markov-decision-process-green?style=for-the-badge)

## Overview

hatata is the stochastic decision optimization engine within the Machinery ecosystem, using Markov Decision Processes and advanced optimization techniques to make optimal health decisions under uncertainty. It integrates evidence from mzekezeke's predictions and diggiden's challenges to provide robust, evidence-based decision making.

**Core Philosophy**: Health decisions involve navigating uncertainty, trade-offs, and multiple competing objectives. By modeling health as a stochastic process with clearly defined states, actions, and utilities, we can make optimal decisions even when faced with incomplete information and adversarial challenges.

The name "hatata" (Swahili for "step by step") reflects the sequential decision-making nature of health optimization, where each step builds upon previous states and influences future possibilities.

## Features

### 🎲 Markov Decision Process Modeling
- **Health state modeling** with discrete and continuous state spaces
- **Action space definition** for health interventions and lifestyle choices
- **Transition probability learning** from individual health patterns
- **Reward function optimization** based on health outcomes

### 🎯 Multi-Objective Utility Optimization
- **Competing goal management** using Pareto optimization
- **Utility function design** with multiple preference models (linear, exponential, sigmoid)
- **Trade-off analysis** between conflicting health objectives
- **Priority-based decision making** with dynamic goal weighting

### 📊 Stochastic Uncertainty Analysis
- **Monte Carlo simulation** for uncertainty quantification
- **Confidence interval estimation** for health projections
- **Risk assessment** using Value-at-Risk and Conditional Value-at-Risk
- **Robustness testing** under various uncertainty scenarios

### 🔗 Evidence Integration
- **Multi-source evidence synthesis** from mzekezeke and diggiden
- **Confidence-weighted aggregation** of predictions and challenges
- **Uncertainty propagation** through decision chains
- **Adaptive learning** from decision outcomes

## Core Components

### HatataCore
Central orchestration engine for stochastic decision making:

```python
from hatata import HatataCore

# Initialize the system
hatata_core = HatataCore(
    stochastic_config={
        "monte_carlo_runs": 1000,
        "confidence_threshold": 0.7,
        "uncertainty_tolerance": 0.15
    }
)

# Add health goals
goal_id = hatata_core.add_health_goal(
    goal_name="Cardiovascular Fitness",
    target_metric="cardiovascular",
    target_value=0.85,
    priority=2.0,
    utility_type=UtilityType.EXPONENTIAL
)

# Make a decision
decision = hatata_core.make_decision(
    current_metrics={
        "cardiovascular": 0.75,
        "stress_level": 0.3,
        "energy_level": 0.6
    },
    time_horizon=30
)
```

### MDPEngine
Markov Decision Process modeling for health states:

```python
from hatata.mdp_engine import MDPEngine, HealthState

# Initialize MDP engine
mdp_engine = MDPEngine()

# Create health state
current_state = HealthState(
    state_id="current_health",
    category=HealthStateCategory.GOOD,
    cardiovascular=0.75,
    immune=0.8,
    nervous=0.7,
    stress_level=0.3
)

# Optimize policy
optimization_results = mdp_engine.optimize_policy(method="value_iteration")

# Get action recommendation
action, expected_utility = mdp_engine.recommend_action(current_state)
```

### UtilityOptimizer
Multi-objective optimization for competing health goals:

```python
from hatata.utility_optimizer import UtilityOptimizer, HealthGoal, UtilityType

# Initialize optimizer
optimizer = UtilityOptimizer(
    optimization_method="pareto",
    aggregation_method="weighted_sum"
)

# Add competing goals
cardio_goal = HealthGoal(
    goal_id="cardio_fitness",
    goal_name="Cardiovascular Fitness",
    target_metric="cardiovascular",
    target_value=0.85,
    priority=2.0,
    utility_type=UtilityType.EXPONENTIAL
)

stress_goal = HealthGoal(
    goal_id="stress_management", 
    goal_name="Stress Management",
    target_metric="stress_level",
    target_value=0.2,  # Lower stress
    priority=1.8,
    utility_type=UtilityType.SIGMOID
)

optimizer.add_goal(cardio_goal)
optimizer.add_goal(stress_goal)

# Optimize multiple objectives
result = optimizer.optimize_multi_objective(
    current_values={"cardiovascular": 0.75, "stress_level": 0.35},
    method="pareto"
)
```

## Stochastic Analysis

### Monte Carlo Simulation
Quantify uncertainty in health projections:

```python
# Analyze decision problem with uncertainty
analysis_results = hatata_core.analyze_decision_problem(
    current_metrics=health_metrics,
    time_horizon=60
)

# Access stochastic analysis
stochastic_results = analysis_results["stochastic_analysis"]
uncertainty_metrics = stochastic_results["uncertainty_metrics"]
risk_assessment = stochastic_results["risk_assessment"]
confidence_intervals = stochastic_results["confidence_intervals"]
```

### Risk Assessment
Comprehensive risk evaluation using stochastic methods:

```python
# Risk metrics for each health dimension
for metric, risk_data in risk_assessment.items():
    print(f"{metric}:")
    print(f"  Risk Score: {risk_data['risk_score']:.3f}")
    print(f"  Probability of Decline: {risk_data['probability_decline']:.1%}")
    print(f"  Value at Risk (5%): {risk_data['value_at_risk_5']:.3f}")
    print(f"  Conditional VaR: {risk_data['conditional_value_at_risk']:.3f}")
```

## Decision Types and Confidence Levels

### Decision Confidence Scoring
Every decision includes comprehensive confidence assessment:

```python
# Decision with confidence metrics
decision = hatata_core.make_decision(current_metrics)

print(f"Recommended Action: {decision.recommended_action}")
print(f"Confidence Score: {decision.confidence_score:.3f}")
print(f"Risk Score: {decision.risk_score:.3f}")
print(f"Robustness Score: {decision.robustness_score:.3f}")

# Evidence sources
for evidence in decision.evidence_sources:
    print(f"Evidence from {evidence.source}: {evidence.confidence_score:.3f}")
```

### Uncertainty Quantification
```python
# Access uncertainty analysis
uncertainty_analysis = decision.uncertainty_analysis
print(f"Evidence Agreement: {uncertainty_analysis['evidence_agreement']:.3f}")
print(f"Total Evidence Sources: {uncertainty_analysis['total_evidence_pieces']}")
print(f"Monte Carlo Validation: {uncertainty_analysis['monte_carlo_validation']}")
```

## Multi-Objective Optimization

### Pareto Frontier Analysis
Find optimal trade-offs between competing objectives:

```python
# Pareto optimization for competing goals
pareto_result = optimizer.optimize_multi_objective(
    current_values=health_metrics,
    method="pareto"
)

# Access Pareto frontier
pareto_solutions = optimizer.pareto_frontier
print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

for i, solution in enumerate(pareto_solutions[:3]):  # Show top 3
    print(f"Solution {i+1}:")
    utilities = solution["solution"]["individual_utilities"]
    for goal, utility in utilities.items():
        print(f"  {goal}: {utility:.3f}")
```

### Goal Conflict Analysis
Identify and resolve conflicts between health objectives:

```python
# Analyze goal conflicts
conflicts = optimizer.analyze_goal_conflicts()

for conflict in conflicts:
    print(f"Conflict: {conflict['goal1']} vs {conflict['goal2']}")
    print(f"Type: {conflict['conflict_type']}")
    print(f"Severity: {conflict['severity']:.2f}")
```

## Installation

```bash
cd hatata
pip install -e .

# For development
pip install -e ".[dev]"

# For advanced optimization features
pip install -e ".[advanced]"

# For research and visualization
pip install -e ".[research]"
```

## Quick Start

```python
from hatata import HatataCore, UtilityType

# Initialize hatata
hatata_core = HatataCore()

# Set up competing health goals
hatata_core.add_health_goal(
    goal_name="Cardiovascular Fitness",
    target_metric="cardiovascular",
    target_value=0.85,
    priority=2.0,
    utility_type=UtilityType.EXPONENTIAL
)

hatata_core.add_health_goal(
    goal_name="Stress Management", 
    target_metric="stress_level",
    target_value=0.2,
    priority=1.8,
    utility_type=UtilityType.SIGMOID
)

# Current health state
current_metrics = {
    "cardiovascular": 0.75,
    "stress_level": 0.35,
    "energy_level": 0.6,
    "immune": 0.8
}

# Make stochastic decision
decision = hatata_core.make_decision(
    current_metrics=current_metrics,
    time_horizon=30
)

print(f"🎯 Recommended Action: {decision.recommended_action}")
print(f"📊 Confidence: {decision.confidence_score:.1%}")
print(f"⚠️  Risk Level: {decision.risk_score:.1%}")
print(f"🛡️  Robustness: {decision.robustness_score:.1%}")

# Show implementation steps
print("\n📋 Implementation Plan:")
for i, step in enumerate(decision.implementation_steps, 1):
    print(f"{i}. {step['description']} (Priority: {step['priority']})")
```

## Advanced Features

### Policy Optimization
Learn optimal health intervention policies:

```python
# Simulate policy performance
initial_states = [create_sample_health_state(wellness) for wellness in [0.6, 0.7, 0.8]]
simulation_results = mdp_engine.simulate_policy_performance(
    initial_states=initial_states,
    simulation_steps=30
)

print(f"Average Total Reward: {simulation_results['average_total_reward']:.2f}")
print(f"Success Rate: {simulation_results['success_rate']:.1%}")
```

### Iterative Decision Making
Demonstrate learning and adaptation over time:

```python
# Simulate multiple decision cycles
for cycle in range(5):
    # Make decision
    decision = hatata_core.make_decision(current_metrics)
    
    # Simulate outcome (simplified)
    if decision.confidence_score > 0.7:
        wellness_change = 0.02  # Positive outcome
    else:
        wellness_change = -0.01  # Negative outcome
    
    # Update health state
    current_metrics["overall_wellness"] += wellness_change
    
    print(f"Cycle {cycle + 1}: Action={decision.recommended_action}, "
          f"Confidence={decision.confidence_score:.2f}, "
          f"New Wellness={current_metrics['overall_wellness']:.2f}")
```

## Integration with Machinery Ecosystem

hatata integrates evidence from all Machinery systems:

```python
# Integration example
def integrated_decision_cycle(health_data):
    # 1. Get predictions from mzekezeke
    mzekezeke_predictions = get_mzekezeke_predictions(health_data)
    
    # 2. Get challenges from diggiden  
    diggiden_challenges = get_diggiden_challenges(mzekezeke_predictions)
    
    # 3. Optimize decision with hatata
    decision = hatata_core.make_decision(
        current_metrics=health_data,
        external_predictions=mzekezeke_predictions,
        adversarial_scenarios=diggiden_challenges
    )
    
    return decision
```

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `stochastic_demo.py`: Complete stochastic decision workflow
- `mdp_optimization.py`: Markov Decision Process modeling
- `utility_optimization.py`: Multi-objective goal optimization
- `uncertainty_analysis.py`: Monte Carlo uncertainty quantification
- `integration_demo.py`: Integration with mzekezeke and diggiden

## Research Foundation

hatata is built on established research in:

- **Markov Decision Processes**: Optimal sequential decision making under uncertainty
- **Multi-Objective Optimization**: Pareto efficiency and trade-off analysis
- **Stochastic Processes**: Monte Carlo methods and uncertainty quantification
- **Decision Theory**: Utility theory and preference modeling
- **Health Economics**: Health utility measurement and QALY frameworks

## Contributing

We welcome contributions in:
- New utility function designs
- Advanced MDP modeling techniques
- Stochastic optimization algorithms
- Multi-objective optimization methods
- Real-world health decision validation

## License

MIT License - Part of the Machinery ecosystem. 
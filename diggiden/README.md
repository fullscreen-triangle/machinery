# diggiden ⚔️

*"The health antagonist"* - Adversarial system for robustness testing and challenge generation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Adversarial](https://img.shields.io/badge/adversarial-testing-red?style=for-the-badge)
![Robustness](https://img.shields.io/badge/robustness-validation-orange?style=for-the-badge)

## Overview

diggiden is the adversarial testing engine within the Machinery ecosystem, designed to continuously challenge health optimization predictions and strategies. Acting as an intelligent antagonist, it tests the robustness of health models by simulating realistic stressors, system failures, and challenging scenarios.

**Core Philosophy**: Health is maintained through constant struggle against entropy, pathogens, stress, and degradation processes. True health optimization must be tested against adversarial conditions to ensure robustness in the real world.

## The Health Balance Insight

diggiden embodies a crucial insight: **health is a complex mixture of things working at different percentages**. A person can feel well even when:
- Cardiovascular system is at 85%
- Immune system is at 90% 
- Nervous system is at 78%
- Metabolic system is at 92%

The key insight: **No system needs to be at 0% for problems to manifest**. Health optimization must account for this complexity and build resilience across multiple systems simultaneously.

## Features

### 🔥 Adversarial Challenge Generation
- **Multi-system stress testing** across cardiovascular, metabolic, immune systems
- **Realistic scenario simulation** based on real-world health challenges
- **Progressive challenge escalation** that adapts to system responses
- **Stealth challenge modes** that test detection capabilities

### ⚖️ Health Balance Modeling
- **Complex system interactions** where multiple systems operate sub-optimally
- **Compensation mechanism testing** to find breaking points
- **Reserve capacity evaluation** under stress conditions
- **System interdependency mapping** for cascading failure analysis

### 🎯 Vulnerability Discovery
- **Weak point identification** in health optimization strategies
- **Attack vector analysis** for health system failures
- **Robustness boundary testing** to find system limits
- **Pattern learning** from successful challenges

### 🧪 Challenge Evolution
- **Adaptive adversarial strategies** that learn from previous challenges
- **Dynamic difficulty adjustment** based on system performance
- **Novel challenge generation** using machine learning
- **Real-world scenario modeling** based on epidemiological data

## Core Components

### AdversarialEngine
Central system for generating and managing health challenges:

```python
from diggiden import AdversarialEngine

engine = AdversarialEngine(
    challenge_intensity="moderate",
    learning_rate=0.1,
    adaptation_enabled=True
)

# Generate a challenge for current health state
challenge = engine.generate_challenge(
    current_health_state=health_data,
    target_systems=["cardiovascular", "metabolic"],
    challenge_type="gradual_degradation"
)

# Evaluate system response
response_analysis = engine.evaluate_response(
    challenge=challenge,
    system_response=optimization_strategy,
    time_window=72  # hours
)
```

### HealthAntagonist
Advanced health balance challenger:

```python
from diggiden.antagonist import HealthAntagonist

antagonist = HealthAntagonist()

# Test health balance under multi-system stress
balance_test = antagonist.challenge_health_balance(
    current_state={
        "cardiovascular": 0.85,
        "immune": 0.90,
        "nervous": 0.78,
        "metabolic": 0.92
    },
    challenge_scenario="work_stress_with_poor_sleep"
)

# Check if person would still feel well
wellness_prediction = antagonist.predict_subjective_wellness(
    system_states=balance_test.resulting_state,
    compensation_effects=balance_test.compensation_analysis
)
```

### Challenge Scenarios
Pre-built realistic health challenges:

```python
from diggiden.challenges import ChallengeScenario

# Metabolic stress challenge
metabolic_challenge = ChallengeScenario.create_metabolic_stress(
    trigger="high_carb_meal_after_poor_sleep",
    intensity=0.7,
    duration=6  # hours
)

# Cardiovascular stress challenge  
cardio_challenge = ChallengeScenario.create_cardio_stress(
    trigger="unexpected_high_intensity_exercise",
    context="dehydrated_state",
    recovery_impediments=["elevated_stress", "inadequate_sleep"]
)

# Multi-system failure simulation
complex_challenge = ChallengeScenario.create_multi_system_failure(
    primary_system="immune",
    secondary_systems=["metabolic", "nervous"],
    cascade_probability=0.8
)
```

## Challenge Strategies

### GRADUAL_DEGRADATION
Slowly increases stress on target systems to test adaptation capacity:

```python
gradual_challenge = engine.create_gradual_challenge(
    target_system="cardiovascular",
    degradation_rate=0.02,  # 2% per day
    duration_days=14,
    detection_threshold=0.05
)
```

### SUDDEN_SHOCK
Tests system response to acute stressors:

```python
shock_challenge = engine.create_sudden_shock(
    shock_type="metabolic_overload",
    intensity=0.9,
    recovery_time_expected=24  # hours
)
```

### MULTI_SYSTEM_FAILURE
Simulates cascading failures across health systems:

```python
cascade_challenge = engine.create_multi_system_failure(
    initial_system="sleep_regulation",
    cascade_pattern="sleep→stress→immune→metabolic",
    failure_probability=0.3
)
```

### STEALTH_PROGRESSION
Hidden challenges that test detection capabilities:

```python
stealth_challenge = engine.create_stealth_challenge(
    hidden_degradation="insulin_sensitivity",
    masking_factors=["exercise_compensation", "dietary_adaptation"],
    revelation_trigger="glucose_tolerance_test"
)
```

## Installation

```bash
cd diggiden
pip install -e .

# For development
pip install -e ".[dev]"

# For advanced challenge modeling
pip install -e ".[advanced]"
```

## Quick Start

```python
from diggiden import AdversarialEngine, HealthAntagonist

# Initialize the adversarial system
engine = AdversarialEngine()
antagonist = HealthAntagonist()

# Create a health state
current_health = {
    "cardiovascular": 0.82,
    "immune": 0.88,
    "nervous": 0.75,
    "metabolic": 0.85,
    "overall_wellness": 0.82
}

# Generate an adversarial challenge
challenge = engine.generate_adaptive_challenge(
    health_state=current_health,
    challenge_goal="test_multi_system_resilience"
)

# Test health balance under challenge
balance_result = antagonist.test_health_balance(
    initial_state=current_health,
    challenge=challenge,
    duration=48  # hours
)

print(f"Challenge: {challenge.description}")
print(f"Systems affected: {challenge.target_systems}")
print(f"Predicted wellness after challenge: {balance_result.final_wellness:.2f}")
print(f"Person would feel: {'well' if balance_result.subjective_wellness > 0.6 else 'unwell'}")
```

## Challenge Philosophy

### Health as Dynamic Balance
diggiden recognizes that health isn't about perfection but about resilient balance:

- **No system is perfect**: All systems operate below 100% efficiency
- **Compensation is key**: Strong systems can compensate for weaker ones
- **Balance points exist**: There are minimum thresholds for subjective wellness
- **Resilience matters**: Ability to maintain balance under stress is crucial

### Adversarial Principles

1. **Realistic Challenges**: Based on actual health stressors and failure modes
2. **Progressive Difficulty**: Challenges adapt to system capabilities
3. **Multi-System Thinking**: Health failures rarely occur in isolation
4. **Learning Opposition**: The system learns from successful challenges

### Challenge Categories

- **Environmental**: Weather, pollution, pathogen exposure
- **Lifestyle**: Poor sleep, dietary stress, sedentary behavior
- **Psychological**: Chronic stress, anxiety, depression
- **Physical**: Overexertion, injury, illness
- **Temporal**: Circadian disruption, seasonal changes

## Integration with Machinery Ecosystem

diggiden challenges predictions from mzekezeke and provides adversarial scenarios for hatata to optimize against:

```python
# Example integration workflow
def integrated_challenge_cycle(health_predictions, optimization_strategy):
    # Challenge the predictions
    challenge = engine.challenge_predictions(health_predictions)
    
    # Test optimization robustness
    robustness_test = engine.test_strategy_robustness(
        strategy=optimization_strategy,
        challenge_scenarios=challenge.scenarios
    )
    
    # Provide feedback to optimization system
    return {
        "challenge_results": challenge,
        "robustness_assessment": robustness_test,
        "recommended_improvements": challenge.adaptation_suggestions
    }
```

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `adversarial_demo.py`: Complete adversarial testing workflow
- `health_balance_demo.py`: Health balance challenge scenarios
- `multi_system_failure.py`: Complex cascading failure simulation
- `challenge_evolution.py`: Adaptive challenge generation

## Research Foundation

diggiden is based on research in:

- **Adversarial Machine Learning**: Robustness testing for AI systems
- **Systems Medicine**: Multi-system health interactions
- **Stress Physiology**: How systems respond to challenges
- **Resilience Theory**: What makes health systems robust

## Contributing

We welcome contributions in:
- New challenge scenarios
- Health system interaction models
- Adversarial strategy improvements
- Real-world validation studies

## License

MIT License - Part of the Machinery ecosystem. 
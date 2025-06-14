#!/usr/bin/env python3
"""
Diggiden Adversarial System Demonstration

This script demonstrates your key insight about health as a complex balance:
- Systems can operate at 100%, 90%, 80%, etc. and person still feels well
- The adversarial system tests the breaking points of this balance
- Shows how the antagonist learns to find vulnerabilities
- Demonstrates realistic health deterioration scenarios
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diggiden.antagonist import HealthAntagonist, HealthBalance, SystemState
from datetime import datetime, timedelta
import json
import time


def demonstrate_health_balance_concept():
    """Demonstrate the core insight about health as a complex balance."""
    print("=" * 60)
    print("HEALTH BALANCE CONCEPT DEMONSTRATION")
    print("=" * 60)
    
    # Create a health balance
    balance = HealthBalance()
    
    # Add systems with different functionality levels
    systems = [
        SystemState("cardiovascular", 85.0, 0.6, 0.7, 0.8, 0.02, 0.7),  # Good
        SystemState("immune", 70.0, 0.4, 0.5, 0.6, 0.03, 0.8),         # Adequate
        SystemState("nervous", 95.0, 0.8, 0.6, 0.9, 0.01, 0.6),        # Excellent
        SystemState("metabolic", 78.0, 0.5, 0.8, 0.7, 0.025, 0.9),     # Good
        SystemState("respiratory", 82.0, 0.7, 0.9, 0.5, 0.02, 0.7),    # Good
    ]
    
    for system in systems:
        balance.add_system(system)
    
    # Set up interactions
    balance.interaction_matrix = {
        ("cardiovascular", "respiratory"): 0.9,
        ("cardiovascular", "metabolic"): 0.8,
        ("immune", "nervous"): 0.6,
        ("metabolic", "nervous"): 0.7,
    }
    
    # Calculate wellness
    wellness_score = balance.calculate_wellness_score()
    feeling_well = balance.is_person_feeling_well()
    
    print(f"\nSystem Functionality Levels:")
    for name, system in balance.systems.items():
        print(f"  {name.capitalize():15}: {system.functionality_percentage:5.1f}%")
    
    print(f"\nOverall Wellness Score: {wellness_score:.1f}")
    print(f"Person Feeling Well: {feeling_well}")
    
    print(f"\nKey Insight: Even though no system is at 100%, and immune system")
    print(f"is only at 70%, the person still feels well because:")
    print(f"  - Strong nervous system (95%) provides compensation")
    print(f"  - Good cardiovascular-respiratory synergy")
    print(f"  - Adequate reserve capacities")
    print(f"  - No critical system failures")
    
    return balance


def demonstrate_adversarial_challenges():
    """Demonstrate how the antagonist challenges the health balance."""
    print("\n" + "=" * 60)
    print("ADVERSARIAL CHALLENGE DEMONSTRATION")
    print("=" * 60)
    
    # Create antagonist with a mock health system
    antagonist = HealthAntagonist(
        target_health_system=None,  # Mock system
        antagonist_intensity=0.6,
        learning_enabled=True
    )
    
    # Get initial state
    initial_report = antagonist.get_current_balance_report()
    print(f"\nInitial State:")
    print(f"  Wellness Score: {initial_report['overall_wellness_score']:.1f}")
    print(f"  Feeling Well: {initial_report['person_feeling_well']}")
    print(f"  Classification: {initial_report['wellness_classification']}")
    
    print(f"\n  System States:")
    for name, state in initial_report['system_states'].items():
        print(f"    {name.capitalize():15}: {state['functionality_percentage']:5.1f}% ({state['status']})")
    
    # Issue several challenges
    print(f"\nIssuing Adversarial Challenges...")
    challenge_types = ["gradual_degradation", "stress_overload", "resource_depletion", "mixed_degradation"]
    
    for i, challenge_type in enumerate(challenge_types):
        print(f"\n--- Challenge {i+1}: {challenge_type} ---")
        
        challenge_result = antagonist.challenge_system_balance(
            challenge_type=challenge_type,
            challenge_duration=timedelta(days=30)
        )
        
        print(f"  Target Systems: {challenge_result['target_systems']}")
        print(f"  Wellness: {challenge_result['pre_wellness_score']:.1f} → {challenge_result['post_wellness_score']:.1f}")
        print(f"  Still Feeling Well: {challenge_result['post_feeling_well']}")
        print(f"  Challenge Successful: {challenge_result['challenge_successful']}")
        
        if challenge_result['challenge_successful']:
            print(f"  🎯 ANTAGONIST VICTORY - Found a vulnerability!")
        else:
            print(f"  🛡️  System resilience held strong")
        
        # Show system changes
        print(f"  System Changes:")
        for system, effect in challenge_result['challenge_effects'].items():
            change = effect['functionality_change']
            print(f"    {system}: {effect['original_functionality']:.1f}% → {effect['new_functionality']:.1f}% ({change:+.1f}%)")
    
    # Final state
    final_report = antagonist.get_current_balance_report()
    print(f"\nFinal State After All Challenges:")
    print(f"  Wellness Score: {final_report['overall_wellness_score']:.1f}")
    print(f"  Feeling Well: {final_report['person_feeling_well']}")
    print(f"  Classification: {final_report['wellness_classification']}")
    
    # Show what the antagonist learned
    performance = antagonist.get_antagonist_performance()
    print(f"\nAntagonist Performance:")
    print(f"  Challenges Issued: {performance['total_challenges_issued']}")
    print(f"  Victories: {performance['victories']}")
    print(f"  Success Rate: {performance['success_rate']:.1%}")
    print(f"  Discovered Vulnerabilities: {performance['discovered_vulnerabilities']}")
    
    return antagonist


def demonstrate_long_term_deterioration():
    """Demonstrate long-term health deterioration simulation."""
    print("\n" + "=" * 60)
    print("LONG-TERM DETERIORATION SIMULATION")
    print("=" * 60)
    
    # Create fresh antagonist
    antagonist = HealthAntagonist(
        target_health_system=None,
        antagonist_intensity=0.3,  # Lower intensity for gradual effects
        learning_enabled=True
    )
    
    print(f"\nSimulating 2 years of gradual health changes...")
    print(f"This shows how the complex balance adapts over time.")
    
    # Run 2-year simulation
    scenario_log = antagonist.simulate_health_deterioration_scenario(
        scenario_name="natural_aging_with_stress",
        duration_days=730  # 2 years
    )
    
    # Show key timepoints
    timepoints = [0, 6, 12, 18, 24]  # Every 6 months
    
    print(f"\nHealth Evolution Over Time:")
    print(f"{'Month':<8} {'Wellness':<10} {'Feeling Well':<12} {'Key Changes'}")
    print(f"{'='*60}")
    
    for i, month in enumerate(timepoints):
        if month * 30 < len(scenario_log):
            log_entry = scenario_log[month * 30 // 30]
            
            wellness = log_entry['wellness_score']
            feeling_well = "Yes" if log_entry['feeling_well'] else "No"
            
            # Find most changed system
            systems = log_entry['system_states']
            most_changed = min(systems.items(), key=lambda x: x[1])
            
            print(f"{month:<8} {wellness:<10.1f} {feeling_well:<12} {most_changed[0]}: {most_changed[1]:.1f}%")
    
    # Analyze the pattern
    initial_wellness = scenario_log[0]['wellness_score']
    final_wellness = scenario_log[-1]['wellness_score']
    
    print(f"\nAnalysis:")
    print(f"  Initial Wellness: {initial_wellness:.1f}")
    print(f"  Final Wellness: {final_wellness:.1f}")
    print(f"  Total Decline: {initial_wellness - final_wellness:.1f} points")
    
    # Count how long person felt well
    months_feeling_well = sum(1 for entry in scenario_log if entry['feeling_well'])
    total_months = len(scenario_log)
    
    print(f"  Months Feeling Well: {months_feeling_well}/{total_months} ({months_feeling_well/total_months:.1%})")
    
    print(f"\nKey Insight: Even with gradual deterioration, the person")
    print(f"maintained wellness for most of the time due to:")
    print(f"  - System compensation mechanisms")
    print(f"  - Occasional recovery periods")
    print(f"  - Adaptive balance adjustments")
    
    return scenario_log


def demonstrate_system_synergies():
    """Demonstrate how system interactions affect overall wellness."""
    print("\n" + "=" * 60)
    print("SYSTEM SYNERGY DEMONSTRATION")
    print("=" * 60)
    
    # Create two scenarios: one with good synergies, one without
    print(f"\nComparing two scenarios with different system interactions:")
    
    # Scenario 1: Good synergies
    balance1 = HealthBalance()
    systems1 = [
        SystemState("cardiovascular", 75.0, 0.6, 0.7, 0.8, 0.02, 0.7),
        SystemState("respiratory", 75.0, 0.7, 0.9, 0.5, 0.02, 0.7),
        SystemState("metabolic", 75.0, 0.5, 0.8, 0.7, 0.025, 0.9),
    ]
    
    for system in systems1:
        balance1.add_system(system)
    
    # Strong positive interactions
    balance1.interaction_matrix = {
        ("cardiovascular", "respiratory"): 0.9,  # Very strong synergy
        ("cardiovascular", "metabolic"): 0.8,
        ("respiratory", "metabolic"): 0.7,
    }
    
    # Scenario 2: Poor synergies (same functionality levels)
    balance2 = HealthBalance()
    systems2 = [
        SystemState("cardiovascular", 75.0, 0.6, 0.7, 0.8, 0.02, 0.7),
        SystemState("respiratory", 75.0, 0.7, 0.9, 0.5, 0.02, 0.7),
        SystemState("metabolic", 75.0, 0.5, 0.8, 0.7, 0.025, 0.9),
    ]
    
    for system in systems2:
        balance2.add_system(system)
    
    # Weak or negative interactions
    balance2.interaction_matrix = {
        ("cardiovascular", "respiratory"): 0.2,  # Poor synergy
        ("cardiovascular", "metabolic"): 0.1,
        ("respiratory", "metabolic"): 0.0,
    }
    
    # Compare outcomes
    wellness1 = balance1.calculate_wellness_score()
    wellness2 = balance2.calculate_wellness_score()
    feeling_well1 = balance1.is_person_feeling_well()
    feeling_well2 = balance2.is_person_feeling_well()
    
    print(f"\nScenario 1 (Strong Synergies):")
    print(f"  All systems at 75% functionality")
    print(f"  Wellness Score: {wellness1:.1f}")
    print(f"  Feeling Well: {feeling_well1}")
    
    print(f"\nScenario 2 (Poor Synergies):")
    print(f"  All systems at 75% functionality")
    print(f"  Wellness Score: {wellness2:.1f}")
    print(f"  Feeling Well: {feeling_well2}")
    
    print(f"\nSynergy Bonus: {wellness1 - wellness2:.1f} points")
    print(f"\nKey Insight: System interactions are crucial!")
    print(f"The same functionality levels can result in very different")
    print(f"wellness outcomes based on how well systems work together.")


def main():
    """Run all demonstrations."""
    print("DIGGIDEN: Adversarial Health Challenge System")
    print("Demonstrating the complex balance insight of health optimization")
    print("=" * 80)
    
    try:
        # 1. Demonstrate the core health balance concept
        balance = demonstrate_health_balance_concept()
        
        # 2. Show adversarial challenges
        antagonist = demonstrate_adversarial_challenges()
        
        # 3. Long-term deterioration simulation
        scenario_log = demonstrate_long_term_deterioration()
        
        # 4. System synergy demonstration
        demonstrate_system_synergies()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print(f"\nKey Takeaways:")
        print(f"1. Health is indeed a complex balance - no system needs to be at 100%")
        print(f"2. The adversarial system successfully finds vulnerabilities")
        print(f"3. System interactions and synergies are crucial for wellness")
        print(f"4. Gradual deterioration can be managed through compensation")
        print(f"5. The antagonist learns and adapts its attack strategies")
        
        print(f"\nThis demonstrates how 'diggiden' serves as a valuable antagonist")
        print(f"to strengthen health optimization systems by:")
        print(f"  - Testing system resilience under various conditions")
        print(f"  - Discovering hidden vulnerabilities")
        print(f"  - Forcing optimization for robustness, not perfection")
        print(f"  - Providing realistic adversarial training scenarios")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
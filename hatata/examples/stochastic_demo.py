#!/usr/bin/env python3
"""
Hatata Stochastic Decision System Demo

This demo showcases the complete Hatata system:
1. Markov Decision Process modeling for health state transitions
2. Utility optimization for multi-objective health goals
3. Stochastic uncertainty analysis with Monte Carlo methods
4. Integration with mzekezeke predictions and diggiden challenges
5. Evidence-based decision making under uncertainty

The demo simulates a person's health journey with competing goals
and shows how Hatata navigates trade-offs and uncertainty.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hatata import HatataCore, HealthGoal, UtilityType
from hatata.mdp_engine import HealthState, HealthStateCategory, HealthAction

def create_sample_health_state(wellness_level: float = 0.7) -> HealthState:
    """Create a sample health state for demonstration."""
    return HealthState(
        state_id=f"demo_state_{datetime.now().strftime('%H%M%S')}",
        category=HealthStateCategory.GOOD,
        cardiovascular=wellness_level + np.random.normal(0, 0.05),
        immune=wellness_level + np.random.normal(0, 0.05),
        nervous=wellness_level + np.random.normal(0, 0.05),
        metabolic=wellness_level + np.random.normal(0, 0.05),
        respiratory=wellness_level + np.random.normal(0, 0.05),
        stress_level=0.3 + np.random.normal(0, 0.05),
        energy_level=wellness_level + np.random.normal(0, 0.05),
        recovery_capacity=wellness_level - 0.1 + np.random.normal(0, 0.05),
        environmental_stress=0.2 + np.random.normal(0, 0.05),
        resource_availability=0.8 + np.random.normal(0, 0.05),
        social_support=0.7 + np.random.normal(0, 0.05)
    )

def setup_health_goals(hatata_core: HatataCore):
    """Set up competing health goals for optimization."""
    print("Setting up health goals...")
    
    # Goal 1: Cardiovascular fitness
    hatata_core.add_health_goal(
        goal_name="Cardiovascular Fitness",
        target_metric="cardiovascular",
        target_value=0.85,
        priority=2.0,
        utility_type=UtilityType.EXPONENTIAL,
        tolerance=0.1,
        time_horizon=60,
        min_acceptable=0.6,
        max_acceptable=1.0,
        utility_params={"decay_rate": 1.5}
    )
    
    # Goal 2: Stress management (competing with fitness - time/energy trade-off)
    hatata_core.add_health_goal(
        goal_name="Stress Management",
        target_metric="stress_level",
        target_value=0.2,  # Lower stress
        priority=1.8,
        utility_type=UtilityType.SIGMOID,
        tolerance=0.15,
        time_horizon=30,
        min_acceptable=0.0,
        max_acceptable=0.5,
        utility_params={"steepness": 2.0}
    )
    
    # Goal 3: Energy optimization
    hatata_core.add_health_goal(
        goal_name="Energy Optimization",
        target_metric="energy_level",
        target_value=0.8,
        priority=1.5,
        utility_type=UtilityType.LOGARITHMIC,
        tolerance=0.1,
        time_horizon=45,
        min_acceptable=0.4,
        max_acceptable=1.0,
        utility_params={"scale": 1.2}
    )
    
    # Goal 4: Immune system strength
    hatata_core.add_health_goal(
        goal_name="Immune System",
        target_metric="immune",
        target_value=0.9,
        priority=2.2,  # High priority
        utility_type=UtilityType.LINEAR,
        tolerance=0.08,
        time_horizon=90,
        min_acceptable=0.5,
        max_acceptable=1.0
    )
    
    print(f"✓ Set up {len(hatata_core.utility_optimizer.goals)} health goals")

def run_stochastic_analysis_demo():
    """Demonstrate comprehensive stochastic health analysis."""
    print("="*60)
    print("HATATA STOCHASTIC DECISION SYSTEM DEMO")
    print("="*60)
    print()
    
    # Initialize Hatata system
    print("1. Initializing Hatata Core System...")
    hatata_core = HatataCore(
        stochastic_config={
            "monte_carlo_runs": 500,  # Reduced for demo speed
            "confidence_threshold": 0.7,
            "uncertainty_tolerance": 0.15
        }
    )
    
    # Set up health goals
    setup_health_goals(hatata_core)
    
    # Create current health state
    print("\n2. Creating Current Health State...")
    current_state = create_sample_health_state(wellness_level=0.65)
    hatata_core.set_current_state(current_state)
    
    print(f"   Current wellness: {current_state.overall_wellness:.2f}")
    print(f"   Cardiovascular: {current_state.cardiovascular:.2f}")
    print(f"   Stress level: {current_state.stress_level:.2f}")
    print(f"   Energy level: {current_state.energy_level:.2f}")
    print(f"   Immune system: {current_state.immune:.2f}")
    
    # Current health metrics
    current_metrics = {
        "cardiovascular": current_state.cardiovascular,
        "stress_level": current_state.stress_level,
        "energy_level": current_state.energy_level,
        "immune": current_state.immune,
        "overall_wellness": current_state.overall_wellness
    }
    
    print("\n3. Running Comprehensive Stochastic Analysis...")
    print("   This includes:")
    print("   • Markov Decision Process optimization")
    print("   • Multi-objective utility analysis") 
    print("   • Monte Carlo uncertainty quantification")
    print("   • Evidence integration and synthesis")
    print()
    
    # Run full analysis
    analysis_start = datetime.now()
    analysis_results = hatata_core.analyze_decision_problem(
        current_metrics=current_metrics,
        time_horizon=60
    )
    analysis_time = (datetime.now() - analysis_start).total_seconds()
    
    print(f"   ✓ Analysis completed in {analysis_time:.2f} seconds")
    
    # Display results
    print("\n4. Analysis Results Summary:")
    print("-" * 40)
    
    # MDP Results
    mdp_results = analysis_results["mdp_analysis"]
    print(f"   MDP Optimal Action: {mdp_results.get('optimal_action', 'None')}")
    print(f"   Expected Utility: {mdp_results.get('expected_utility', 0):.3f}")
    
    # Utility Optimization Results
    utility_results = analysis_results["utility_analysis"]
    if "multi_objective_result" in utility_results:
        mo_result = utility_results["multi_objective_result"]
        print(f"   Multi-objective Success: {mo_result.get('success', False)}")
        print(f"   Aggregate Utility: {mo_result.get('aggregate_utility', 0):.3f}")
        
        if "optimal_values" in mo_result:
            print("   Optimal Health Metrics:")
            for metric, value in mo_result["optimal_values"].items():
                current_val = current_metrics.get(metric, 0)
                change = value - current_val
                direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
                print(f"     {metric}: {current_val:.2f} {direction} {value:.2f}")
    
    # Stochastic Analysis Results
    stochastic_results = analysis_results["stochastic_analysis"]
    print(f"   Monte Carlo Runs: {stochastic_results.get('monte_carlo_runs', 0)}")
    
    uncertainty_metrics = stochastic_results.get("uncertainty_metrics", {})
    if uncertainty_metrics:
        print("   Uncertainty Analysis:")
        for metric, metrics_data in uncertainty_metrics.items():
            unc_coeff = metrics_data.get("uncertainty_coefficient", 0)
            print(f"     {metric}: {unc_coeff:.3f} uncertainty coefficient")
    
    # Risk Assessment
    risk_assessment = stochastic_results.get("risk_assessment", {})
    if risk_assessment:
        print("   Risk Assessment:")
        for metric, risk_data in risk_assessment.items():
            risk_score = risk_data.get("risk_score", 0)
            prob_decline = risk_data.get("probability_decline", 0)
            print(f"     {metric}: {risk_score:.3f} risk score, {prob_decline:.1%} decline probability")
    
    # Final Decision
    final_decision = analysis_results["final_recommendation"]
    print(f"\n5. Final Decision:")
    print("-" * 40)
    print(f"   Recommended Action: {final_decision.recommended_action}")
    print(f"   Decision Confidence: {final_decision.confidence_score:.3f}")
    print(f"   Risk Score: {final_decision.risk_score:.3f}")
    print(f"   Robustness Score: {final_decision.robustness_score:.3f}")
    print(f"   Expected Utility: {final_decision.expected_utility:.3f}")
    
    # Evidence sources
    print(f"\n   Evidence Sources ({len(final_decision.evidence_sources)}):")
    for evidence in final_decision.evidence_sources:
        print(f"     • {evidence.source}: {evidence.confidence_score:.3f} confidence")
    
    # Implementation steps
    print(f"\n   Implementation Steps:")
    for i, step in enumerate(final_decision.implementation_steps, 1):
        print(f"     {i}. {step.get('description', step.get('action', 'Unknown'))}")
        print(f"        Priority: {step.get('priority', 'medium')}, Timeline: {step.get('timeline', 'unknown')}")
    
    return hatata_core, analysis_results, final_decision

def demonstrate_uncertainty_visualization(hatata_core: HatataCore, analysis_results: dict):
    """Create visualizations of uncertainty analysis."""
    print("\n6. Uncertainty Visualization:")
    print("-" * 40)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hatata Stochastic Health Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confidence Intervals Plot
        confidence_intervals = analysis_results["stochastic_analysis"].get("confidence_intervals", {})
        if confidence_intervals:
            ax = axes[0, 0]
            metrics = list(confidence_intervals.keys())[:4]  # Limit to 4 metrics
            
            for i, metric in enumerate(metrics):
                intervals = confidence_intervals[metric]
                if "p50" in intervals:  # Check if we have percentile data
                    time_points = range(len(intervals["p50"]))
                    ax.fill_between(time_points, intervals.get("p25", []), intervals.get("p75", []), 
                                  alpha=0.3, label=f"{metric} (25-75%)")
                    ax.plot(time_points, intervals["p50"], label=f"{metric} median", linewidth=2)
            
            ax.set_title('Health Metric Projections with Uncertainty')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Health Metric Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Risk Assessment Heat Map
        risk_assessment = analysis_results["stochastic_analysis"].get("risk_assessment", {})
        if risk_assessment:
            ax = axes[0, 1]
            
            metrics = list(risk_assessment.keys())
            risk_types = ["risk_score", "probability_decline", "probability_severe_decline"]
            
            # Create risk matrix
            risk_matrix = []
            for metric in metrics:
                risk_row = []
                for risk_type in risk_types:
                    risk_row.append(risk_assessment[metric].get(risk_type, 0))
                risk_matrix.append(risk_row)
            
            if risk_matrix:
                im = ax.imshow(risk_matrix, cmap='Reds', aspect='auto')
                ax.set_title('Risk Assessment Matrix')
                ax.set_xticks(range(len(risk_types)))
                ax.set_xticklabels([rt.replace('_', ' ').title() for rt in risk_types], rotation=45)
                ax.set_yticks(range(len(metrics)))
                ax.set_yticklabels(metrics)
                
                # Add text annotations
                for i in range(len(metrics)):
                    for j in range(len(risk_types)):
                        text = ax.text(j, i, f'{risk_matrix[i][j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
                
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 3. Utility Function Landscapes
        ax = axes[1, 0]
        
        # Create utility landscape for one metric
        current_metrics = {
            "cardiovascular": 0.65,
            "stress_level": 0.35,
            "energy_level": 0.6,
            "immune": 0.7
        }
        
        if hatata_core.utility_optimizer.goals:
            metric = "cardiovascular"  # Focus on cardiovascular
            landscape = hatata_core.utility_optimizer.get_utility_landscape(
                current_metrics, metric, (0.3, 1.0), num_points=50
            )
            
            values = landscape["values"]
            utilities = landscape["utilities"]
            current_value = landscape["current_value"]
            optimal_value = landscape["optimal_value"]
            
            ax.plot(values, utilities, 'b-', linewidth=2, label='Utility Function')
            ax.axvline(current_value, color='red', linestyle='--', label=f'Current ({current_value:.2f})')
            ax.axvline(optimal_value, color='green', linestyle='--', label=f'Optimal ({optimal_value:.2f})')
            ax.fill_between(values, utilities, alpha=0.3)
            
            ax.set_title(f'Utility Landscape: {metric.title()}')
            ax.set_xlabel('Metric Value')
            ax.set_ylabel('Utility Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Decision Evidence Comparison
        ax = axes[1, 1]
        
        final_decision = analysis_results["final_recommendation"]
        evidence_sources = final_decision.evidence_sources
        
        if evidence_sources:
            sources = [ev.source for ev in evidence_sources]
            confidences = [ev.confidence_score for ev in evidence_sources]
            
            bars = ax.bar(sources, confidences, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(sources)])
            ax.set_title('Evidence Source Confidence')
            ax.set_ylabel('Confidence Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"hatata_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   ✓ Uncertainty visualization saved as: {plot_filename}")
        
        # Show the plot
        plt.show()
    
    except ImportError:
        print("   ! Matplotlib/Seaborn not available - skipping visualization")
    except Exception as e:
        print(f"   ! Visualization error: {e}")

def demonstrate_decision_iteration(hatata_core: HatataCore):
    """Demonstrate iterative decision making over time."""
    print("\n7. Iterative Decision Making Simulation:")
    print("-" * 40)
    
    # Simulate 5 time steps with evolving health state
    simulation_results = []
    current_wellness = 0.65
    
    for step in range(5):
        print(f"\n   Step {step + 1}:")
        
        # Create current state
        current_state = create_sample_health_state(current_wellness)
        hatata_core.set_current_state(current_state)
        
        current_metrics = {
            "cardiovascular": current_state.cardiovascular,
            "stress_level": current_state.stress_level,
            "energy_level": current_state.energy_level,
            "immune": current_state.immune,
            "overall_wellness": current_state.overall_wellness
        }
        
        # Make decision
        decision = hatata_core.make_decision(current_metrics, time_horizon=30)
        
        print(f"     Action: {decision.recommended_action}")
        print(f"     Confidence: {decision.confidence_score:.3f}")
        print(f"     Risk: {decision.risk_score:.3f}")
        
        # Simulate outcome (simplified)
        if decision.confidence_score > 0.7:
            # High confidence decision -> positive outcome
            wellness_change = np.random.normal(0.02, 0.01)
        elif decision.confidence_score > 0.5:
            # Medium confidence -> neutral outcome
            wellness_change = np.random.normal(0.0, 0.015)
        else:
            # Low confidence -> potential negative outcome
            wellness_change = np.random.normal(-0.01, 0.02)
        
        current_wellness = max(0.3, min(1.0, current_wellness + wellness_change))
        
        simulation_results.append({
            "step": step + 1,
            "wellness": current_wellness,
            "action": decision.recommended_action,
            "confidence": decision.confidence_score,
            "risk": decision.risk_score
        })
    
    # Show simulation summary
    print(f"\n   Simulation Summary:")
    print(f"     Initial wellness: {simulation_results[0]['wellness']:.3f}")
    print(f"     Final wellness: {simulation_results[-1]['wellness']:.3f}")
    print(f"     Average confidence: {np.mean([r['confidence'] for r in simulation_results]):.3f}")
    print(f"     Average risk: {np.mean([r['risk'] for r in simulation_results]):.3f}")
    
    return simulation_results

def demonstrate_system_integration():
    """Demonstrate integration with mzekezeke and diggiden systems."""
    print("\n8. System Integration Capabilities:")
    print("-" * 40)
    
    print("   Hatata is designed to integrate with:")
    print("   • mzekezeke: ML predictions and scientific health analysis")
    print("     - Provides predictive features for MDP state transitions")
    print("     - Enhances utility function parameterization")
    print("     - Supplies evidence for stochastic decision making")
    print()
    print("   • diggiden: Adversarial challenges and robustness testing")
    print("     - Tests decision robustness under adversarial conditions")
    print("     - Provides worst-case scenario analysis")
    print("     - Enhances uncertainty quantification")
    print()
    print("   • Machinery: Temporal dynamics and system orchestration")
    print("     - Coordinates multi-system evidence integration")
    print("     - Manages temporal health state evolution")
    print("     - Provides contextual decision framework")
    print()
    print("   Integration Benefits:")
    print("   ✓ Multi-layered evidence validation")
    print("   ✓ Adversarial robustness testing")
    print("   ✓ Scientific prediction validation")
    print("   ✓ Comprehensive uncertainty modeling")

def main():
    """Run the complete Hatata demonstration."""
    try:
        # Main demonstration
        hatata_core, analysis_results, final_decision = run_stochastic_analysis_demo()
        
        # Uncertainty visualization
        demonstrate_uncertainty_visualization(hatata_core, analysis_results)
        
        # Iterative decision making
        simulation_results = demonstrate_decision_iteration(hatata_core)
        
        # System integration overview
        demonstrate_system_integration()
        
        # Final summary
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        
        quality_metrics = hatata_core.get_decision_quality_metrics()
        print(f"Total decisions made: {quality_metrics.get('total_decisions', 0)}")
        print(f"Average confidence: {quality_metrics.get('average_confidence', 0):.3f}")
        print(f"Average robustness: {quality_metrics.get('average_robustness', 0):.3f}")
        
        print("\nHatata successfully demonstrated:")
        print("✓ Stochastic health decision making")
        print("✓ Multi-objective utility optimization") 
        print("✓ Markov Decision Process modeling")
        print("✓ Monte Carlo uncertainty quantification")
        print("✓ Evidence integration and synthesis")
        print("✓ Iterative decision refinement")
        print("✓ System integration readiness")
        
        print(f"\nFinal recommendation confidence: {final_decision.confidence_score:.3f}")
        print(f"Decision robustness score: {final_decision.robustness_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
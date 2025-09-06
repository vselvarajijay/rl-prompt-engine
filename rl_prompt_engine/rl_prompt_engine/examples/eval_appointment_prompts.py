#!/usr/bin/env python3
"""
Evaluation and Testing Tools for Appointment Booking Meta-Prompting System

This module provides comprehensive evaluation tools to:
1. Test prompt effectiveness across different scenarios
2. Compare different prompt strategies
3. Analyze system performance and learning progress
4. Generate detailed reports and insights
"""

import argparse
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from stable_baselines3 import PPO
from appointment_prompt_env import AppointmentPromptEnv
from appointment_prompt_generator import AppointmentPromptGenerator, AppointmentPromptDatabase

class AppointmentPromptEvaluator:
    """Comprehensive evaluator for appointment booking prompt system."""
    
    def __init__(self, model_path: str = "ppo_appointment_prompts"):
        self.model_path = model_path
        self.model = None
        self.generator = AppointmentPromptGenerator()
        self.database = AppointmentPromptDatabase()
        
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def evaluate_comprehensive(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation across all scenarios."""
        if not self.model:
            if not self.load_model():
                return {}
        
        print(f"🧪 Running comprehensive evaluation with {num_episodes} episodes...")
        
        # Test across all customer types, stages, and urgency levels
        customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
        conversation_stages = ["early", "middle", "late", "closing"]
        urgency_levels = ["low", "medium", "high"]
        
        results = []
        component_usage = Counter()
        effectiveness_by_scenario = defaultdict(list)
        
        for episode in range(num_episodes):
            # Randomly select scenario
            customer_type = np.random.randint(0, 6)
            conversation_stage = np.random.randint(0, 4)
            urgency_level = np.random.randint(0, 3)
            
            # Generate prompt
            env = AppointmentPromptEnv()
            obs, _ = env.reset()
            obs["customer_type"][0] = customer_type
            obs["conversation_stage"][0] = conversation_stage
            obs["urgency_level"][0] = urgency_level
            
            selected_components = []
            step = 0
            max_steps = 6
            
            while step < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                # Decode flattened action
                action = int(action[0]) if hasattr(action, '__len__') else int(action)
                finish = action % 2
                position = (action // 2) % 6  # MAX_PROMPT_LENGTH
                component_idx = (action // (2 * 6)) % 10  # N_PROMPT_COMPONENTS
                
                if finish == 1:
                    break
                
                if component_idx not in selected_components and position < 6:
                    selected_components.append(component_idx)
                    obs["prompt_so_far"][position] = 1.0
                    obs["turn"][0] = step + 1
                
                step += 1
            
            # Calculate effectiveness
            template = self.generator.generate_prompt_template(
                customer_type=customer_type,
                conversation_stage=conversation_stage,
                urgency_level=urgency_level,
                selected_components=selected_components,
                customer_psychology=obs["customer_psychology"]
            )
            
            # Record results
            result = {
                "episode": episode,
                "customer_type": customer_types[customer_type],
                "conversation_stage": conversation_stages[conversation_stage],
                "urgency_level": urgency_levels[urgency_level],
                "effectiveness": template.effectiveness_score,
                "components_used": len(selected_components),
                "component_names": [list(self.generator.config["prompt_components"].keys())[i] for i in selected_components],
                "efficiency": len(selected_components) / max(step, 1)
            }
            
            results.append(result)
            
            # Track component usage
            for comp_idx in selected_components:
                comp_name = list(self.generator.config["prompt_components"].keys())[comp_idx]
                component_usage[comp_name] += 1
            
            # Track effectiveness by scenario
            scenario_key = f"{customer_types[customer_type]}_{conversation_stages[conversation_stage]}_{urgency_levels[urgency_level]}"
            effectiveness_by_scenario[scenario_key].append(template.effectiveness_score)
        
        # Calculate statistics
        effectiveness_scores = [r["effectiveness"] for r in results]
        component_counts = [r["components_used"] for r in results]
        efficiency_scores = [r["efficiency"] for r in results]
        
        stats = {
            "total_episodes": num_episodes,
            "avg_effectiveness": np.mean(effectiveness_scores),
            "std_effectiveness": np.std(effectiveness_scores),
            "max_effectiveness": np.max(effectiveness_scores),
            "min_effectiveness": np.min(effectiveness_scores),
            "avg_components": np.mean(component_counts),
            "avg_efficiency": np.mean(efficiency_scores),
            "component_usage": dict(component_usage),
            "effectiveness_by_scenario": {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} 
                                        for k, v in effectiveness_by_scenario.items()},
            "results": results
        }
        
        return stats
    
    def evaluate_by_customer_type(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate performance by customer type."""
        if not self.model:
            if not self.load_model():
                return {}
        
        customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
        results_by_type = {}
        
        for customer_type in range(6):
            print(f"🧪 Testing {customer_types[customer_type]} customers...")
            
            effectiveness_scores = []
            component_usage = Counter()
            
            for episode in range(num_episodes):
                # Random conversation stage and urgency
                conversation_stage = np.random.randint(0, 4)
                urgency_level = np.random.randint(0, 3)
                
                # Generate prompt
                env = AppointmentPromptEnv()
                obs, _ = env.reset()
                obs["customer_type"][0] = customer_type
                obs["conversation_stage"][0] = conversation_stage
                obs["urgency_level"][0] = urgency_level
                
                selected_components = []
                step = 0
                max_steps = 6
                
                while step < max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    # Decode flattened action
                    action = int(action[0]) if hasattr(action, '__len__') else int(action)
                    finish = action % 2
                    position = (action // 2) % 6  # MAX_PROMPT_LENGTH
                    component_idx = (action // (2 * 6)) % 10  # N_PROMPT_COMPONENTS
                    
                    if finish == 1:
                        break
                    
                    if component_idx not in selected_components and position < 6:
                        selected_components.append(component_idx)
                        obs["prompt_so_far"][position] = 1.0
                        obs["turn"][0] = step + 1
                    
                    step += 1
                
                # Calculate effectiveness
                template = self.generator.generate_prompt_template(
                    customer_type=customer_type,
                    conversation_stage=conversation_stage,
                    urgency_level=urgency_level,
                    selected_components=selected_components,
                    customer_psychology=obs["customer_psychology"]
                )
                
                effectiveness_scores.append(template.effectiveness_score)
                
                # Track component usage
                for comp_idx in selected_components:
                    comp_name = list(self.generator.config["prompt_components"].keys())[comp_idx]
                    component_usage[comp_name] += 1
            
            results_by_type[customer_types[customer_type]] = {
                "avg_effectiveness": np.mean(effectiveness_scores),
                "std_effectiveness": np.std(effectiveness_scores),
                "max_effectiveness": np.max(effectiveness_scores),
                "min_effectiveness": np.min(effectiveness_scores),
                "component_usage": dict(component_usage),
                "episodes": num_episodes
            }
        
        return results_by_type
    
    def compare_strategies(self, num_episodes: int = 30) -> Dict[str, Any]:
        """Compare different prompt strategies."""
        if not self.model:
            if not self.load_model():
                return {}
        
        print("🔍 Comparing different prompt strategies...")
        
        # Define different strategies
        strategies = {
            "rapport_focused": [0, 1, 9],  # rapport_building, needs_assessment, personalization
            "value_focused": [2, 3, 6],    # value_proposition, objection_handling, incentive_offering
            "urgency_focused": [4, 7, 6],  # urgency_creation, appointment_booking, incentive_offering
            "social_proof_focused": [5, 0, 1],  # social_proof, rapport_building, needs_assessment
            "balanced": [0, 1, 2, 3, 7]   # mix of components
        }
        
        strategy_results = {}
        
        for strategy_name, component_indices in strategies.items():
            print(f"  Testing {strategy_name} strategy...")
            
            effectiveness_scores = []
            
            for episode in range(num_episodes):
                # Random scenario
                customer_type = np.random.randint(0, 6)
                conversation_stage = np.random.randint(0, 4)
                urgency_level = np.random.randint(0, 3)
                
                # Use fixed strategy
                selected_components = component_indices[:min(len(component_indices), 6)]
                
                # Calculate effectiveness
                template = self.generator.generate_prompt_template(
                    customer_type=customer_type,
                    conversation_stage=conversation_stage,
                    urgency_level=urgency_level,
                    selected_components=selected_components,
                    customer_psychology=np.random.random(5)  # Random psychology
                )
                
                effectiveness_scores.append(template.effectiveness_score)
            
            strategy_results[strategy_name] = {
                "avg_effectiveness": np.mean(effectiveness_scores),
                "std_effectiveness": np.std(effectiveness_scores),
                "max_effectiveness": np.max(effectiveness_scores),
                "min_effectiveness": np.min(effectiveness_scores),
                "components": [list(self.generator.config["prompt_components"].keys())[i] for i in selected_components]
            }
        
        return strategy_results
    
    def generate_report(self, stats: Dict[str, Any]) -> str:
        """Generate a detailed evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("APPOINTMENT BOOKING META-PROMPTING SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall performance
        report.append("📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Episodes: {stats['total_episodes']}")
        report.append(f"Average Effectiveness: {stats['avg_effectiveness']:.3f} ± {stats['std_effectiveness']:.3f}")
        report.append(f"Maximum Effectiveness: {stats['max_effectiveness']:.3f}")
        report.append(f"Minimum Effectiveness: {stats['min_effectiveness']:.3f}")
        report.append(f"Average Components Used: {stats['avg_components']:.1f}")
        report.append(f"Average Efficiency: {stats['avg_efficiency']:.3f}")
        report.append("")
        
        # Component usage
        report.append("🎯 COMPONENT USAGE FREQUENCY")
        report.append("-" * 40)
        sorted_components = sorted(stats['component_usage'].items(), key=lambda x: x[1], reverse=True)
        for component, count in sorted_components:
            percentage = (count / stats['total_episodes']) * 100
            report.append(f"{component:<25}: {count:3d} ({percentage:5.1f}%)")
        report.append("")
        
        # Scenario performance
        report.append("📈 PERFORMANCE BY SCENARIO")
        report.append("-" * 40)
        sorted_scenarios = sorted(stats['effectiveness_by_scenario'].items(), 
                                key=lambda x: x[1]['mean'], reverse=True)
        for scenario, data in sorted_scenarios[:10]:  # Top 10 scenarios
            report.append(f"{scenario:<35}: {data['mean']:.3f} ± {data['std']:.3f} ({data['count']} episodes)")
        report.append("")
        
        # Best and worst scenarios
        all_scenarios = [(k, v) for k, v in stats['effectiveness_by_scenario'].items()]
        best_scenario = max(all_scenarios, key=lambda x: x[1]['mean'])
        worst_scenario = min(all_scenarios, key=lambda x: x[1]['mean'])
        
        report.append("🏆 BEST PERFORMING SCENARIO")
        report.append("-" * 40)
        report.append(f"Scenario: {best_scenario[0]}")
        report.append(f"Effectiveness: {best_scenario[1]['mean']:.3f} ± {best_scenario[1]['std']:.3f}")
        report.append(f"Episodes: {best_scenario[1]['count']}")
        report.append("")
        
        report.append("⚠️  WORST PERFORMING SCENARIO")
        report.append("-" * 40)
        report.append(f"Scenario: {worst_scenario[0]}")
        report.append(f"Effectiveness: {worst_scenario[1]['mean']:.3f} ± {worst_scenario[1]['std']:.3f}")
        report.append(f"Episodes: {worst_scenario[1]['count']}")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if stats['avg_effectiveness'] < 0.7:
            report.append("• Overall effectiveness is below 0.7 - consider retraining the model")
        
        if stats['std_effectiveness'] > 0.2:
            report.append("• High variance in effectiveness - model may need more training data")
        
        if stats['avg_components'] < 3:
            report.append("• Low component usage - prompts may be too simple")
        elif stats['avg_components'] > 5:
            report.append("• High component usage - prompts may be too complex")
        
        # Find underperforming scenarios
        underperforming = [(k, v) for k, v in stats['effectiveness_by_scenario'].items() 
                          if v['mean'] < stats['avg_effectiveness'] - 0.1]
        if underperforming:
            report.append(f"• {len(underperforming)} scenarios are underperforming - focus training on these")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate appointment booking prompt system")
    parser.add_argument("--model", type=str, default="ppo_appointment_prompts", help="Model path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_report.txt", help="Output report file")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive evaluation")
    parser.add_argument("--by-customer-type", action="store_true", help="Evaluate by customer type")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare different strategies")
    
    args = parser.parse_args()
    
    print("🧪 Appointment Booking Meta-Prompting System Evaluation")
    print("=" * 60)
    
    evaluator = AppointmentPromptEvaluator(args.model)
    
    if args.comprehensive or (not args.by_customer_type and not args.compare_strategies):
        print("Running comprehensive evaluation...")
        stats = evaluator.evaluate_comprehensive(args.episodes)
        
        if stats:
            print("\n📊 EVALUATION RESULTS:")
            print(f"Average Effectiveness: {stats['avg_effectiveness']:.3f}")
            print(f"Max Effectiveness: {stats['max_effectiveness']:.3f}")
            print(f"Min Effectiveness: {stats['min_effectiveness']:.3f}")
            print(f"Average Components: {stats['avg_components']:.1f}")
            
            # Generate and save report
            report = evaluator.generate_report(stats)
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Detailed report saved to: {args.output}")
    
    if args.by_customer_type:
        print("\nEvaluating by customer type...")
        results = evaluator.evaluate_by_customer_type(args.episodes // 6)
        
        print("\n📊 RESULTS BY CUSTOMER TYPE:")
        print("-" * 50)
        for customer_type, data in results.items():
            print(f"{customer_type:<15}: {data['avg_effectiveness']:.3f} ± {data['std_effectiveness']:.3f}")
    
    if args.compare_strategies:
        print("\nComparing strategies...")
        results = evaluator.compare_strategies(args.episodes // 5)
        
        print("\n📊 STRATEGY COMPARISON:")
        print("-" * 50)
        for strategy, data in results.items():
            print(f"{strategy:<20}: {data['avg_effectiveness']:.3f} ± {data['std_effectiveness']:.3f}")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()

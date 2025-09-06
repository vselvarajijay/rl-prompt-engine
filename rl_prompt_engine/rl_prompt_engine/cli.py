#!/usr/bin/env python3
"""
Command Line Interface for Meta-Prompting System
"""

import argparse
import json
from .core.meta_prompting import MetaPromptingSystem, create_system, train_system

def main():
    parser = argparse.ArgumentParser(description="Meta-Prompting System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL model")
    train_parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps")
    train_parser.add_argument("--save-path", default="ppo_meta_prompting", help="Path to save model")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate prompt templates")
    generate_parser.add_argument("--model-path", help="Path to trained model")
    generate_parser.add_argument("--customer-type", type=int, choices=range(6), default=0, help="Customer type (0-5)")
    generate_parser.add_argument("--conversation-stage", "--stage", type=int, choices=range(4), default=0, help="Conversation stage (0-3)")
    generate_parser.add_argument("--urgency-level", "--urgency", type=int, choices=range(3), default=0, help="Urgency level (0-2)")
    generate_parser.add_argument("--output", default="template.txt", help="Output file")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate templates for multiple scenarios")
    batch_parser.add_argument("--model-path", help="Path to trained model")
    batch_parser.add_argument("--scenarios", help="JSON file with scenarios")
    batch_parser.add_argument("--output", default="templates.json", help="Output file")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with predefined scenarios")
    demo_parser.add_argument("--model-path", help="Path to trained model")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show detailed help and examples")
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("🚀 Training RL model...")
        system = train_system(args.timesteps, args.save_path)
        print("✅ Training completed!")
        
    elif args.command == "generate":
        print("🎯 Generating prompt template...")
        
        # Use default model path if none specified
        model_path = args.model_path or "ppo_meta_prompting"
        system = create_system(model_path)
        
        try:
            template = system.generate_prompt_template(
                args.customer_type, 
                args.conversation_stage, 
                args.urgency_level
            )
            
            with open(args.output, "w") as f:
                f.write(template)
            print(f"✅ Template saved to {args.output}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("💡 You need to train a model first. Run:")
            print("   poetry run python -m rl_prompt_engine.cli train --timesteps 1000")
        
    elif args.command == "batch":
        print("📊 Generating templates for multiple scenarios...")
        model_path = args.model_path or "ppo_meta_prompting"
        system = create_system(model_path)
        
        if args.scenarios:
            with open(args.scenarios, "r") as f:
                scenarios = json.load(f)
        else:
            # Default scenarios
            scenarios = [
                {"customer_type": 0, "conversation_stage": 0, "urgency_level": 0, "name": "Cautious Early"},
                {"customer_type": 1, "conversation_stage": 1, "urgency_level": 1, "name": "Price Shopper Middle"},
                {"customer_type": 2, "conversation_stage": 3, "urgency_level": 2, "name": "Ready Buyer Closing"},
                {"customer_type": 5, "conversation_stage": 2, "urgency_level": 0, "name": "Skeptical Late"},
            ]
        
        templates = system.generate_templates_for_scenarios(scenarios)
        system.save_templates(templates, args.output)
        
    elif args.command == "demo":
        print("🎭 Running demo...")
        model_path = args.model_path or "ppo_meta_prompting"
        system = create_system(model_path)
        
        # Demo scenarios
        scenarios = [
            {"customer_type": 0, "conversation_stage": 0, "urgency_level": 0, "name": "Cautious Early"},
            {"customer_type": 2, "conversation_stage": 3, "urgency_level": 2, "name": "Ready Buyer Closing"},
        ]
        
        templates = system.generate_templates_for_scenarios(scenarios)
        
        for i, template_data in enumerate(templates):
            print(f"\n--- Template {i+1}: {template_data['scenario']['name']} ---")
            print(template_data["template"][:200] + "...")
        
        system.save_templates(templates, "demo_templates.json")
        print("✅ Demo completed! Templates saved to demo_templates.json")
        
    elif args.command == "help":
        print("🎯 RL Prompt Engine CLI Help")
        print("=" * 40)
        print()
        print("📋 Workflow:")
        print("1. First, train a model (required)")
        print("2. Then generate templates using the trained model")
        print()
        print("Available commands:")
        print()
        print("1. Train a model (REQUIRED FIRST):")
        print("   poetry run python -m rl_prompt_engine.cli train --timesteps 1000")
        print()
        print("2. Generate a single template:")
        print("   poetry run python -m rl_prompt_engine.cli generate --customer-type 0 --stage 0 --urgency 0")
        print()
        print("3. Generate multiple templates:")
        print("   poetry run python -m rl_prompt_engine.cli batch")
        print()
        print("4. Run demo:")
        print("   poetry run python -m rl_prompt_engine.cli demo")
        print()
        print("📊 Parameters:")
        print("Customer Types: 0=cautious, 1=price_shopper, 2=ready_buyer, 3=research_buyer, 4=impulse_buyer, 5=skeptical")
        print("Stages: 0=early, 1=middle, 2=late, 3=closing")
        print("Urgency: 0=low, 1=medium, 2=high")
        print()
        print("💡 Quick start:")
        print("   poetry run python -m rl_prompt_engine.cli train --timesteps 1000")
        print("   poetry run python -m rl_prompt_engine.cli generate --customer-type 0 --stage 0 --urgency 0")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simplified Command Line Interface for RL Prompt Engine
"""

import argparse
import json
from .core.prompt_system import PromptSystem, create_system, train_system
from .core.logging_config import setup_logging, get_logger

def main():
    # Setup logging for CLI
    loggers = setup_logging()
    cli_logger = loggers['training']
    
    parser = argparse.ArgumentParser(description="RL Prompt Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL model")
    train_parser.add_argument("--config", default="configs/generic_config.json", help="Path to configuration file")
    train_parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps")
    train_parser.add_argument("--save-path", default="models/ppo_prompt_system", help="Path to save model")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate prompt templates")
    generate_parser.add_argument("--config", default="configs/generic_config.json", help="Path to configuration file")
    generate_parser.add_argument("--model-path", help="Path to trained model")
    generate_parser.add_argument("--context-type", type=int, default=0, help="Context type index")
    generate_parser.add_argument("--conversation-stage", "--stage", type=int, default=0, help="Conversation stage index")
    generate_parser.add_argument("--urgency-level", "--urgency", type=int, default=0, help="Urgency level index")
    generate_parser.add_argument("--output", default="template.txt", help="Output file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available context types, stages, and components")
    list_parser.add_argument("--config", default="configs/generic_config.json", help="Path to configuration file")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Generate configuration from prompt")
    config_parser.add_argument("--description", required=True, help="Natural language description of your business scenario")
    config_parser.add_argument("--output", default="configs/custom_config.json", help="Output configuration file")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cli_logger.info(f"Starting training with config: {args.config}, timesteps: {args.timesteps}")
        print("🚀 Training RL model...")
        system = PromptSystem(config_file=args.config)
        system.train(args.timesteps, args.save_path)
        print("✅ Training completed!")
        
    elif args.command == "generate":
        print("🎯 Generating prompt template...")
        
        # Use default model path if none specified
        model_path = args.model_path or "models/ppo_prompt_system"
        
        try:
            system = PromptSystem(model_path=model_path, config_file=args.config)
            template = system.generate_meta_prompt(
                context_type=args.context_type,
                conversation_stage=args.conversation_stage,
                urgency_level=args.urgency_level
            )
            
            with open(args.output, "w") as f:
                f.write(template)
            print(f"✅ Template saved to {args.output}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("💡 You need to train a model first. Run:")
            print("   poetry run python -m rl_prompt_engine.cli train --timesteps 1000")
            
    elif args.command == "list":
        print("📋 Available configuration options:")
        try:
            system = PromptSystem(config_file=args.config)
            context_info = system.get_context_info()
            
            print("\n🎯 Context Types:")
            for i, context_type in enumerate(context_info["context_types"]):
                print(f"  {i}: {context_type}")
            
            print("\n📈 Conversation Stages:")
            for i, stage in enumerate(context_info["conversation_stages"]):
                print(f"  {i}: {stage}")
            
            print("\n⚡ Urgency Levels:")
            for i, urgency in enumerate(context_info["urgency_levels"]):
                print(f"  {i}: {urgency}")
            
            print("\n🧩 Prompt Components:")
            for i, component in enumerate(context_info["prompt_components"]):
                print(f"  {i}: {component}")
                
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            
    elif args.command == "config":
        print("🤖 Generating configuration from description...")
        try:
            from .core.config_generator import ConfigGenerator
            generator = ConfigGenerator()
            config = generator.generate_config(
                description=args.description,
                output_file=args.output
            )
            print(f"✅ Configuration saved to {args.output}")
        except Exception as e:
            print(f"❌ Error generating config: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simplified Command Line Interface for RL Prompt Engine
"""

import argparse
import json
from .core.prompt_engine import PromptEngine
from .core.logging_config import setup_logging

def main():
    # Setup logging for CLI
    loggers = setup_logging()
    cli_logger = loggers['training']
    
    parser = argparse.ArgumentParser(description="RL Prompt Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the PPO model")
    train_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    train_parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps")
    train_parser.add_argument("--save-path", default="models/prompt_engine_model", help="Path to save model")
    train_parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate prompt templates")
    generate_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    generate_parser.add_argument("--model-path", help="Path to trained model")
    generate_parser.add_argument("--context-type", type=int, default=0, help="Context type index")
    generate_parser.add_argument("--conversation-stage", "--stage", type=int, default=0, help="Conversation stage index")
    generate_parser.add_argument("--urgency-level", "--urgency", type=int, default=0, help="Urgency level index")
    generate_parser.add_argument("--custom-vars", required=True, help="Custom variables as JSON string (required)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available context types, stages, and components")
    list_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a prompt strategy")
    eval_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    eval_parser.add_argument("--model-path", help="Path to trained model")
    eval_parser.add_argument("--context-type", type=int, default=0, help="Context type index")
    eval_parser.add_argument("--conversation-stage", "--stage", type=int, default=0, help="Conversation stage index")
    eval_parser.add_argument("--urgency-level", "--urgency", type=int, default=0, help="Urgency level index")
    eval_parser.add_argument("--strategy", nargs="+", help="List of component names to evaluate")
    
    # Template command
    template_parser = subparsers.add_parser("template", help="Template management")
    template_subparsers = template_parser.add_subparsers(dest="template_action", help="Template actions")
    
    # List templates
    template_list_parser = template_subparsers.add_parser("list", help="List available templates")
    template_list_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    
    # Validate template
    template_validate_parser = template_subparsers.add_parser("validate", help="Validate a template")
    template_validate_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    template_validate_parser.add_argument("--template", default="meta_prompt_template", help="Template name to validate")
    
    # Show template
    template_show_parser = template_subparsers.add_parser("show", help="Show template content")
    template_show_parser.add_argument("--config", default="rl_prompt_engine/configs/generic_config.json", help="Path to configuration file")
    template_show_parser.add_argument("--template", default="meta_prompt_template", help="Template name to show")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cli_logger.info(f"Starting training with config: {args.config}, timesteps: {args.timesteps}")
        print("🚀 Training PPO model...")
        
        engine = PromptEngine(config_file=args.config)
        engine.train(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        print("✅ Training completed!")
        
    elif args.command == "generate":
        print("🎯 Generating prompt template...")
        
        # Use default model path if none specified
        model_path = args.model_path or "models/prompt_engine_model"
        
        # Parse custom variables (required)
        try:
            custom_variables = json.loads(args.custom_vars)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing custom variables: {e}")
            return
        
        try:
            # Load engine and generate template
            engine = PromptEngine(config_file=args.config)
            engine.load_model(model_path)
            
            template = engine.generate_template(
                context_type=args.context_type,
                conversation_stage=args.conversation_stage,
                urgency_level=args.urgency_level,
                custom_variables=custom_variables
            )
            
            print("✅ Generated Prompt Template:")
            print("=" * 60)
            print(template)
            print("=" * 60)
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("💡 You need to train a model first. Run:")
            print("   python -m rl_prompt_engine.cli train --timesteps 1000")
            
    elif args.command == "list":
        print("📋 Available configuration options:")
        try:
            engine = PromptEngine(config_file=args.config)
            context_info = engine.get_available_contexts()
            
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
            
    elif args.command == "evaluate":
        print("📊 Evaluating prompt strategy...")
        
        # Use default model path if none specified
        model_path = args.model_path or "models/prompt_engine_model"
        
        try:
            engine = PromptEngine(config_file=args.config)
            engine.load_model(model_path)
            
            if not args.strategy:
                print("❌ Please provide a strategy with --strategy")
                return
            
            evaluation = engine.evaluate_strategy(
                context_type=args.context_type,
                conversation_stage=args.conversation_stage,
                urgency_level=args.urgency_level,
                strategy=args.strategy
            )
            
            print(f"📈 Evaluation Results:")
            print(f"  Total Reward: {evaluation['total_reward']:.3f}")
            print(f"  Final Reward: {evaluation['final_reward']:.3f}")
            print(f"  Component Count: {evaluation['component_count']}")
            print(f"  Effectiveness: {evaluation['effectiveness']:.3f}")
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("💡 You need to train a model first. Run:")
            print("   python -m rl_prompt_engine.cli train --timesteps 1000")
    
    elif args.command == "template":
        engine = PromptEngine(args.config)
        
        if args.template_action == "list":
            templates = engine.get_available_templates()
            print("📋 Available templates:")
            for template in templates:
                print(f"   - {template}")
            if not templates:
                print("   No templates found")
        
        elif args.template_action == "validate":
            try:
                validation = engine.validate_template(args.template)
                print(f"✅ Template '{args.template}' validation:")
                print(f"   Variables found: {len(validation['found_variables'])}")
                for var in validation['found_variables']:
                    print(f"     - {var}")
            except FileNotFoundError:
                print(f"❌ Template '{args.template}' not found")
        
        elif args.template_action == "show":
            try:
                content = engine.load_template(args.template)
                print(f"📄 Template '{args.template}' content:")
                print("=" * 50)
                print(content)
                print("=" * 50)
            except FileNotFoundError:
                print(f"❌ Template '{args.template}' not found")
        
        else:
            template_parser.print_help()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

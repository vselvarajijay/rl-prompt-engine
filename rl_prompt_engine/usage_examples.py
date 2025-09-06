#!/usr/bin/env python3
"""
Usage Examples for RL Prompt Engine

This script demonstrates how to use the rl_prompt_engine package
for generating appointment booking prompts.

Before running, make sure to:
1. Copy env.template to .env: cp env.template .env
2. Add your OpenAI API key to .env: OPENAI_API_KEY=your-key-here
"""

import os
from dotenv import load_dotenv
from rl_prompt_engine import MetaPromptingSystem, create_system, train_system

# Load environment variables
load_dotenv()

def example_1_basic_usage():
    """Example 1: Basic usage with a pre-trained system"""
    print("🔹 Example 1: Basic Usage")
    print("=" * 50)
    
    # Create a new system
    system = MetaPromptingSystem()
    
    # Generate a prompt template for a cautious customer in early conversation
    template = system.generate_prompt_template(
        customer_type=0,  # cautious
        conversation_stage=0,  # early
        urgency_level=0  # low
    )
    
    print("Generated Template:")
    print(template)
    print()

def example_2_training_system():
    """Example 2: Training a new system"""
    print("🔹 Example 2: Training a New System")
    print("=" * 50)
    
    # Create and train a new system
    print("Training RL model... (this may take a few minutes)")
    system = train_system(total_timesteps=1000, save_path="my_trained_model")
    
    # Use the trained system
    template = system.generate_prompt_template(
        customer_type=2,  # ready_buyer
        conversation_stage=2,  # late
        urgency_level=2  # high
    )
    
    print("Generated Template:")
    print(template)
    print()

def example_3_multiple_scenarios():
    """Example 3: Generating templates for multiple scenarios"""
    print("🔹 Example 3: Multiple Scenarios")
    print("=" * 50)
    
    system = MetaPromptingSystem()
    
    # Define different customer scenarios
    scenarios = [
        {
            "customer_type": 0,  # cautious
            "conversation_stage": 0,  # early
            "urgency_level": 0,  # low
            "description": "Cautious customer, early stage, low urgency"
        },
        {
            "customer_type": 1,  # price_shopper
            "conversation_stage": 1,  # middle
            "urgency_level": 1,  # medium
            "description": "Price shopper, middle stage, medium urgency"
        },
        {
            "customer_type": 2,  # ready_buyer
            "conversation_stage": 3,  # closing
            "urgency_level": 2,  # high
            "description": "Ready buyer, closing stage, high urgency"
        }
    ]
    
    # Generate templates for all scenarios
    templates = system.generate_templates_for_scenarios(scenarios)
    
    for i, template_data in enumerate(templates, 1):
        print(f"Scenario {i}: {template_data['scenario']['description']}")
        print("-" * 30)
        print(template_data['template'])
        print()

def example_4_custom_customer_context():
    """Example 4: Using custom customer context"""
    print("🔹 Example 4: Custom Customer Context")
    print("=" * 50)
    
    system = MetaPromptingSystem()
    
    # Generate with custom context
    template = system.generate_prompt_template(
        customer_type=5,  # skeptical
        conversation_stage=1,  # middle
        urgency_level=1,  # medium
        customer_context={
            "budget": "$25,000",
            "timeline": "next month",
            "previous_concerns": "reliability issues with last car"
        }
    )
    
    print("Generated Template with Custom Context:")
    print(template)
    print()

def example_5_save_and_load_templates():
    """Example 5: Saving and loading templates"""
    print("🔹 Example 5: Save and Load Templates")
    print("=" * 50)
    
    system = MetaPromptingSystem()
    
    # Generate templates for different scenarios
    scenarios = [
        {"customer_type": 0, "conversation_stage": 0, "urgency_level": 0},
        {"customer_type": 1, "conversation_stage": 1, "urgency_level": 1},
        {"customer_type": 2, "conversation_stage": 2, "urgency_level": 2}
    ]
    
    templates = system.generate_templates_for_scenarios(scenarios)
    
    # Save templates to file
    system.save_templates(templates, "my_templates.json")
    
    # Load templates from file
    loaded_templates = system.load_templates("my_templates.json")
    
    print(f"✅ Saved and loaded {len(loaded_templates)} templates")
    print()

def example_6_using_cli():
    """Example 6: Using the Command Line Interface"""
    print("🔹 Example 6: Command Line Interface")
    print("=" * 50)
    
    print("You can also use the CLI:")
    print()
    print("1. Train a model:")
    print("   poetry run python -m rl_prompt_engine.cli train --timesteps 5000")
    print()
    print("2. Generate a template:")
    print("   poetry run python -m rl_prompt_engine.cli generate --customer-type 0 --stage 0 --urgency 0")
    print()
    print("3. Generate multiple templates:")
    print("   poetry run python -m rl_prompt_engine.cli generate-batch --scenarios scenarios.json")
    print()

if __name__ == "__main__":
    print("🎯 RL Prompt Engine Usage Examples")
    print("=" * 60)
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   Some examples may not work without it.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or add it to your .env file")
        print()
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_training_system()
        example_3_multiple_scenarios()
        example_4_custom_customer_context()
        example_5_save_and_load_templates()
        example_6_using_cli()
        
        print("🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have installed the package with: poetry install")

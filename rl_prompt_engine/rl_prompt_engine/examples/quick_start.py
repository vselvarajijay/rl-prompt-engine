#!/usr/bin/env python3
"""
Quick Start Example

This example shows how to use the meta-prompting system
to generate prompt templates for appointment booking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_prompting import MetaPromptingSystem

def main():
    print("🚀 Meta-Prompting System - Quick Start")
    print("=" * 50)
    
    # Create system
    print("1. Creating meta-prompting system...")
    system = MetaPromptingSystem()
    
    # Train model (small training for demo)
    print("2. Training RL model...")
    system.train(total_timesteps=5000, save_path="demo_model")
    
    # Generate templates for different scenarios
    print("3. Generating prompt templates...")
    
    scenarios = [
        {
            "customer_type": 0,  # cautious
            "conversation_stage": 0,  # early
            "urgency_level": 0,  # low
            "name": "Cautious Early Customer"
        },
        {
            "customer_type": 2,  # ready_buyer
            "conversation_stage": 3,  # closing
            "urgency_level": 2,  # high
            "name": "Ready Buyer Closing"
        }
    ]
    
    templates = system.generate_templates_for_scenarios(scenarios)
    
    # Display results
    print("\n📋 Generated Templates:")
    print("=" * 50)
    
    for i, template_data in enumerate(templates):
        print(f"\n--- Template {i+1}: {template_data['scenario']['name']} ---")
        print(template_data["template"][:300] + "...")
        print("-" * 50)
    
    # Save templates
    system.save_templates(templates, "quick_start_templates.json")
    print(f"\n✅ Generated {len(templates)} templates!")
    print("💾 Templates saved to quick_start_templates.json")
    
    print("\n🎯 Usage Example:")
    print("=" * 50)
    print("1. Use the generated templates as system prompts for any LLM")
    print("2. Fill in the parameters with actual customer data")
    print("3. LLM will generate personalized appointment booking messages")
    
    # Show parameter example
    print("\n📝 Parameter Example:")
    print("Template: 'Hi {first_name}! I hope you're having a great day.'")
    print("Filled:   'Hi Sarah! I hope you're having a great day.'")

if __name__ == "__main__":
    main()

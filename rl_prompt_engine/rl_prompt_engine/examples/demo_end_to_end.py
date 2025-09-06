#!/usr/bin/env python3
"""
Demo script for End-to-End Prompt Generation (RL + OpenAI)

This script demonstrates the complete pipeline:
1. RL agent determines optimal prompt strategy
2. OpenAI generates the actual prompt
3. Results are stored and can be retrieved
"""

import os
import json
from openai_prompt_generator import OpenAIPromptGenerator

def demo_end_to_end():
    """Demonstrate the end-to-end prompt generation system."""
    print("🚀 END-TO-END PROMPT GENERATION DEMO")
    print("   (RL Agent + OpenAI GPT)")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        return
    
    print("✅ OpenAI API key found")
    
    try:
        # Initialize generator
        print("\n🔧 Initializing RL + OpenAI generator...")
        generator = OpenAIPromptGenerator(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        print("✅ Generator ready!")
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Cautious Customer - Early Stage",
                "customer_type": 0,  # cautious
                "conversation_stage": 0,  # early
                "urgency_level": 0,  # low
                "customer_context": {
                    "name": "Sarah",
                    "interested_in": "Honda Civic",
                    "budget": "$25,000",
                    "concerns": "reliability and fuel economy"
                }
            },
            {
                "name": "Ready Buyer - Closing Stage",
                "customer_type": 2,  # ready_buyer
                "conversation_stage": 3,  # closing
                "urgency_level": 2,  # high
                "customer_context": {
                    "name": "Mike",
                    "interested_in": "Ford F-150",
                    "budget": "$45,000",
                    "needs_car_by": "next week",
                    "has_financing": True
                }
            },
            {
                "name": "Price Shopper - Middle Stage",
                "customer_type": 1,  # price_shopper
                "conversation_stage": 1,  # middle
                "urgency_level": 1,  # medium
                "customer_context": {
                    "name": "Jennifer",
                    "interested_in": "Toyota Camry",
                    "budget": "$30,000",
                    "comparing_dealers": True,
                    "wants_best_deal": True
                }
            }
        ]
        
        print(f"\n📝 Generating prompts for {len(scenarios)} scenarios...")
        print("This demonstrates the complete RL + OpenAI pipeline:")
        print("1. RL agent analyzes customer context")
        print("2. RL agent selects optimal prompt strategy")
        print("3. OpenAI generates professional prompt")
        print("4. Results are stored in database")
        
        results = generator.batch_generate_prompts(scenarios)
        
        print(f"\n📊 DEMO RESULTS:")
        print("=" * 80)
        
        for i, (scenario, result) in enumerate(zip(scenarios, results)):
            print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            print(f"Customer Context: {scenario['customer_context']['name']} - {scenario['customer_context']['interested_in']}")
            print(f"RL Strategy: {', '.join(result['strategy']['selected_components'])}")
            print(f"Effectiveness Score: {result['strategy']['effectiveness_score']:.3f}")
            print(f"OpenAI Model: {result['model_used']}")
            print(f"\nGenerated META-PROMPT (for another AI to use):")
            print("-" * 60)
            print(result['generated_prompt'])
            print("-" * 60)
        
        print(f"\n✅ SUCCESS! Generated {len(results)} professional prompts")
        print("💾 All prompts saved to database")
        
        # Show database statistics
        print(f"\n📈 DATABASE STATISTICS:")
        stats = generator.database.get_statistics()
        print(f"Total prompts: {stats['total_templates']}")
        print(f"Average effectiveness: {stats['avg_effectiveness']:.3f}")
        print(f"Max effectiveness: {stats['max_effectiveness']:.3f}")
        
        print(f"\n🎯 KEY BENEFITS DEMONSTRATED:")
        print("✅ RL agent learns optimal strategies for different customer types")
        print("✅ OpenAI generates natural, professional prompts")
        print("✅ System adapts to conversation stage and urgency level")
        print("✅ Prompts are context-aware and personalized")
        print("✅ Results are stored for future reference and analysis")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Run: python end_to_end_interface.py (for full interface)")
        print("2. Customize scenarios for your specific use case")
        print("3. Train RL agent with your own data")
        print("4. Integrate with your CRM or booking system")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure you have a valid OpenAI API key and the RL model is trained.")

def main():
    """Main demo function."""
    try:
        demo_end_to_end()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

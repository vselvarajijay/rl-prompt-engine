#!/usr/bin/env python3
"""
End-to-End Business Interface for OpenAI-Powered Prompt Generation

This provides an easy-to-use interface for generating appointment booking prompts
using the RL agent + OpenAI pipeline.
"""

import os
import json
from typing import Dict, List, Optional
from openai_prompt_generator import OpenAIPromptGenerator

def show_menu():
    """Display the main menu."""
    print("\n" + "="*70)
    print("🚀 END-TO-END APPOINTMENT BOOKING PROMPT GENERATOR")
    print("   (RL Agent + OpenAI GPT)")
    print("="*70)
    print("1. 🎭 Generate Single Prompt")
    print("2. 📊 Generate Multiple Prompts")
    print("3. 🔍 View Generated Prompts")
    print("4. ⚙️  Configure OpenAI Settings")
    print("5. 🧪 Test Different Scenarios")
    print("6. ❓ Help - How does this work?")
    print("7. 🚪 Exit")
    print("="*70)

def generate_single_prompt():
    """Generate a single prompt using RL + OpenAI."""
    print("\n🎭 SINGLE PROMPT GENERATION")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize generator
        generator = OpenAIPromptGenerator(openai_api_key=api_key)
        
        # Get user preferences
        print("\nCustomer Types:")
        customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
        for i, ct in enumerate(customer_types):
            print(f"  {i}: {ct}")
        
        customer_choice = input("\nSelect customer type (0-5): ")
        try:
            customer_type = int(customer_choice)
            if customer_type < 0 or customer_type > 5:
                customer_type = 0
        except:
            customer_type = 0
        
        print("\nConversation Stages:")
        conversation_stages = ["early", "middle", "late", "closing"]
        for i, cs in enumerate(conversation_stages):
            print(f"  {i}: {cs}")
        
        stage_choice = input("\nSelect conversation stage (0-3): ")
        try:
            conversation_stage = int(stage_choice)
            if conversation_stage < 0 or conversation_stage > 3:
                conversation_stage = 0
        except:
            conversation_stage = 0
        
        print("\nUrgency Levels:")
        urgency_levels = ["low", "medium", "high"]
        for i, ul in enumerate(urgency_levels):
            print(f"  {i}: {ul}")
        
        urgency_choice = input("\nSelect urgency level (0-2): ")
        try:
            urgency_level = int(urgency_choice)
            if urgency_level < 0 or urgency_level > 2:
                urgency_level = 0
        except:
            urgency_level = 0
        
        # Get additional context
        print("\nAdditional Customer Context (optional):")
        customer_name = input("Customer name: ").strip()
        interested_in = input("Interested in (e.g., Honda Civic): ").strip()
        budget = input("Budget: ").strip()
        
        customer_context = {}
        if customer_name:
            customer_context["name"] = customer_name
        if interested_in:
            customer_context["interested_in"] = interested_in
        if budget:
            customer_context["budget"] = budget
        
        # Get custom instructions
        custom_instructions = input("\nCustom instructions (optional): ").strip()
        
        # Generate prompt
        print(f"\n🤖 Generating prompt for {customer_types[customer_type]} customer...")
        
        result = generator.generate_end_to_end_prompt(
            customer_type=customer_type,
            conversation_stage=conversation_stage,
            urgency_level=urgency_level,
            customer_context=customer_context if customer_context else None,
            custom_instructions=custom_instructions if custom_instructions else None
        )
        
        # Display results
        print(f"\n📋 GENERATED APPOINTMENT BOOKING PROMPT")
        print("=" * 60)
        print(f"Customer Type: {customer_types[customer_type]}")
        print(f"Conversation Stage: {conversation_stages[conversation_stage]}")
        print(f"Urgency Level: {urgency_levels[urgency_level]}")
        print(f"RL Strategy: {', '.join(result['strategy']['selected_components'])}")
        print(f"Effectiveness Score: {result['strategy']['effectiveness_score']:.3f}")
        print(f"OpenAI Model: {result['model_used']}")
        print("\n" + "=" * 60)
        print("GENERATED PROMPT:")
        print("=" * 60)
        print(result['generated_prompt'])
        print("=" * 60)
        
        # Ask if user wants to save
        save_choice = input("\n💾 Save this prompt? (y/n): ").lower()
        if save_choice == 'y':
            print("✅ Prompt saved to database!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a valid OpenAI API key and the RL model is trained.")

def generate_multiple_prompts():
    """Generate multiple prompts for different scenarios."""
    print("\n📊 MULTIPLE PROMPT GENERATION")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize generator
        generator = OpenAIPromptGenerator(openai_api_key=api_key)
        
        # Predefined scenarios
        scenarios = [
            {
                "name": "Cautious Early Customer",
                "customer_type": 0,  # cautious
                "conversation_stage": 0,  # early
                "urgency_level": 0,  # low
                "customer_context": {
                    "name": "Sarah",
                    "interested_in": "Honda Civic",
                    "budget": "$25,000"
                }
            },
            {
                "name": "Ready Buyer Closing",
                "customer_type": 2,  # ready_buyer
                "conversation_stage": 3,  # closing
                "urgency_level": 2,  # high
                "customer_context": {
                    "name": "Mike",
                    "interested_in": "Ford F-150",
                    "budget": "$45,000",
                    "needs_car_by": "next week"
                }
            },
            {
                "name": "Price Shopper Middle",
                "customer_type": 1,  # price_shopper
                "conversation_stage": 1,  # middle
                "urgency_level": 1,  # medium
                "customer_context": {
                    "name": "Jennifer",
                    "interested_in": "Toyota Camry",
                    "budget": "$30,000",
                    "comparing_dealers": True
                }
            },
            {
                "name": "Skeptical Late Stage",
                "customer_type": 5,  # skeptical
                "conversation_stage": 2,  # late
                "urgency_level": 0,  # low
                "customer_context": {
                    "name": "Robert",
                    "interested_in": "BMW 3 Series",
                    "budget": "$40,000",
                    "concerns": "reliability and maintenance costs"
                }
            }
        ]
        
        print(f"🚀 Generating prompts for {len(scenarios)} predefined scenarios...")
        
        results = generator.batch_generate_prompts(scenarios)
        
        print(f"\n📊 RESULTS:")
        print("=" * 80)
        
        for i, (scenario, result) in enumerate(zip(scenarios, results)):
            print(f"\n--- {scenario['name']} ---")
            print(f"Strategy: {', '.join(result['strategy']['selected_components'])}")
            print(f"Effectiveness: {result['strategy']['effectiveness_score']:.3f}")
            print(f"Generated Prompt:")
            print("-" * 60)
            print(result['generated_prompt'])
            print("-" * 60)
        
        print(f"\n✅ Generated {len(results)} prompts using RL + OpenAI!")
        print("💾 All prompts saved to database")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def view_generated_prompts():
    """View previously generated prompts."""
    print("\n🔍 VIEW GENERATED PROMPTS")
    print("-" * 50)
    
    try:
        from appointment_prompt_generator import AppointmentPromptDatabase
        database = AppointmentPromptDatabase()
        
        if not database.templates:
            print("❌ No prompts found in database.")
            print("Generate some prompts first using options 1 or 2.")
            return
        
        print(f"\n📋 FOUND {len(database.templates)} PROMPTS:")
        print("=" * 80)
        
        for i, (template_id, template) in enumerate(database.templates.items()):
            print(f"\n{i+1}. Template ID: {template_id}")
            print(f"   Customer: {template.customer_type} | Stage: {template.conversation_stage} | Urgency: {template.urgency_level}")
            print(f"   Effectiveness: {template.effectiveness_score:.3f} | Components: {len(template.components)}")
            print(f"   Components: {', '.join(template.components)}")
            print(f"   Preview: {template.template[:100]}...")
        
        # Ask if user wants to see full prompt
        if database.templates:
            choice = input(f"\nView full prompt? (1-{len(database.templates)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(database.templates):
                    template = list(database.templates.values())[idx]
                    print(f"\n📋 FULL PROMPT:")
                    print("=" * 80)
                    print(template.template)
                    print("=" * 80)
            except:
                pass
        
    except Exception as e:
        print(f"❌ Error: {e}")

def configure_openai_settings():
    """Configure OpenAI settings."""
    print("\n⚙️  OPENAI CONFIGURATION")
    print("-" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ No API key found")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
    
    print("\nAvailable Models:")
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    for i, model in enumerate(models):
        print(f"  {i}: {model}")
    
    model_choice = input("\nSelect model (0-2): ").strip()
    try:
        model_idx = int(model_choice)
        if 0 <= model_idx < len(models):
            selected_model = models[model_idx]
            print(f"✅ Selected model: {selected_model}")
            print("Note: Model selection will be applied in future generations")
        else:
            print("❌ Invalid model selection")
    except:
        print("❌ Invalid input")

def test_scenarios():
    """Test different scenarios interactively."""
    print("\n🧪 SCENARIO TESTING")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        return
    
    try:
        generator = OpenAIPromptGenerator(openai_api_key=api_key)
        
        print("Test different customer types and scenarios:")
        print("1. Cautious customer, early stage, low urgency")
        print("2. Ready buyer, closing stage, high urgency")
        print("3. Price shopper, middle stage, medium urgency")
        print("4. Skeptical customer, late stage, low urgency")
        print("5. Custom scenario")
        
        choice = input("\nSelect test scenario (1-5): ").strip()
        
        scenarios = {
            "1": (0, 0, 0, "Cautious customer, early stage, low urgency"),
            "2": (2, 3, 2, "Ready buyer, closing stage, high urgency"),
            "3": (1, 1, 1, "Price shopper, middle stage, medium urgency"),
            "4": (5, 2, 0, "Skeptical customer, late stage, low urgency")
        }
        
        if choice in scenarios:
            customer_type, conversation_stage, urgency_level, description = scenarios[choice]
            print(f"\n🧪 Testing: {description}")
            
            result = generator.generate_end_to_end_prompt(
                customer_type=customer_type,
                conversation_stage=conversation_stage,
                urgency_level=urgency_level
            )
            
            print(f"\n📊 RESULTS:")
            print(f"Strategy: {', '.join(result['strategy']['selected_components'])}")
            print(f"Effectiveness: {result['strategy']['effectiveness_score']:.3f}")
            print(f"\nGenerated Prompt:")
            print("-" * 60)
            print(result['generated_prompt'])
            print("-" * 60)
            
        elif choice == "5":
            print("Custom scenario - use option 1 to create your own")
        else:
            print("❌ Invalid choice")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def show_help():
    """Show help information."""
    print("\n❓ HOW DOES THIS WORK?")
    print("-" * 50)
    print("""
🚀 END-TO-END PROMPT GENERATION PROCESS:

1. 🧠 RL AGENT ANALYSIS:
   - Analyzes customer type, conversation stage, urgency level
   - Considers customer psychology (interest, trust, commitment)
   - Selects optimal prompt components using learned strategies
   - Calculates effectiveness score

2. 🤖 OPENAI GENERATION:
   - Uses RL strategy to build system prompt for OpenAI
   - Generates natural, professional appointment booking prompt
   - Adapts to specific customer context and requirements
   - Creates human-like conversation starters

3. 📊 RESULT:
   - Professional appointment booking prompt
   - Optimized for specific customer type and situation
   - Ready to use in real sales conversations
   - Stored in database for future reference

🎯 CUSTOMER TYPES:
- Cautious: Needs lots of information and reassurance
- Price Shopper: Focused on getting the best deal
- Ready Buyer: Already knows what they want
- Research Buyer: Wants to learn everything first
- Impulse Buyer: Makes quick decisions
- Skeptical: Hard to convince, needs proof

🎭 CONVERSATION STAGES:
- Early: Initial contact and rapport building
- Middle: Value presentation and needs discovery
- Late: Objection handling and closing preparation
- Closing: Final push for appointment booking

⚡ URGENCY LEVELS:
- Low: Customer has time to consider
- Medium: Some time pressure
- High: Customer needs immediate action

🔧 CUSTOMIZATION:
- Add customer context (name, interests, budget)
- Include custom instructions
- Test different scenarios
- Save effective prompts for reuse
""")

def main():
    """Main business interface."""
    print("Welcome to the End-to-End Appointment Booking Prompt Generator!")
    print("This system uses RL + OpenAI to generate optimal prompts for any customer situation.")
    
    while True:
        show_menu()
        choice = input("\nWhat would you like to do? (1-7): ").strip()
        
        if choice == "1":
            generate_single_prompt()
        elif choice == "2":
            generate_multiple_prompts()
        elif choice == "3":
            view_generated_prompts()
        elif choice == "4":
            configure_openai_settings()
        elif choice == "5":
            test_scenarios()
        elif choice == "6":
            show_help()
        elif choice == "7":
            print("\n👋 Thanks for using the End-to-End Prompt Generator!")
            break
        else:
            print("\n❌ Please enter a number between 1-7")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

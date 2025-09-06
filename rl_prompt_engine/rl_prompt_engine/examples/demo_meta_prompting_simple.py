#!/usr/bin/env python3
"""
Simple Demo of Meta-Prompting System (RL Agent generates prompts for other LLMs)

This demonstrates the correct approach without requiring a trained model:
1. RL agent determines optimal prompt strategy
2. System generates META-PROMPTS that instruct other LLMs
3. Meta-prompts are used to generate actual appointment booking messages
"""

from appointment_prompt_generator import AppointmentPromptGenerator
import numpy as np

def generate_meta_prompt_example(customer_type, conversation_stage, urgency_level, selected_components, customer_context=None):
    """Generate a meta-prompt example based on RL strategy using few-shot prompting."""
    
    customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
    conversation_stages = ["early", "middle", "late", "closing"]
    urgency_levels = ["low", "medium", "high"]
    
    customer_type_name = customer_types[customer_type]
    stage_name = conversation_stages[conversation_stage]
    urgency_name = urgency_levels[urgency_level]
    
    # Generate few-shot examples based on strategy components
    examples = []
    
    if "rapport_building" in selected_components and "needs_assessment" in selected_components:
        examples.append(f"""Example 1: "Hi {customer_context.get('first_name', 'Sarah')}! I hope you're having a great day. I'm calling from {customer_context.get('dealership_name', 'Metro Honda')} about the {customer_context.get('car_model', 'Honda Civic')} you've been researching. I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car?" """)
    
    if "value_proposition" in selected_components and "urgency_creation" in selected_components:
        examples.append(f"""Example 2: "The {customer_context.get('car_model', 'Ford F-150')} offers excellent value with {customer_context.get('key_benefit', 'powerful towing capacity and durability')}, and we have limited availability this week. I can get you a special financing rate of 2.9% and $500 cash back. Would you be available for a 1-hour appointment today or tomorrow?" """)
    
    if "social_proof" in selected_components and "incentive_offering" in selected_components:
        examples.append(f"""Example 3: "Many customers like you have been very satisfied with their {customer_context.get('car_model', 'Toyota Camry')} purchase at {customer_context.get('dealership_name', 'City Toyota')}. The {customer_context.get('car_model', 'Toyota Camry')} offers excellent value with {customer_context.get('key_benefit', 'long-term value and resale')}. I can get you a special financing rate of 3.2% and extended warranty. Can we schedule a 45-minute appointment this week?" """)
    
    if "objection_handling" in selected_components and "personalization" in selected_components:
        examples.append(f"""Example 4: "I understand your concerns about reliability and maintenance costs with the {customer_context.get('car_model', 'BMW 3 Series')}. Many customers like you have been very satisfied with their {customer_context.get('car_model', 'BMW 3 Series')} purchase at {customer_context.get('dealership_name', 'Luxury Motors')}. I'd love to understand what you're looking for in your next vehicle and address any concerns you might have. Would you be available for a 1-hour appointment when convenient?" """)
    
    # If no examples match, create a generic one
    if not examples:
        examples.append(f"""Example 1: "Hi {customer_context.get('first_name', 'Sarah')}! I hope you're having a great day. I'm calling from {customer_context.get('dealership_name', 'our dealership')} about the {customer_context.get('car_model', 'car')} you've been researching. I'd love to help you find the perfect vehicle that meets your needs and budget. Would you be available for a 30-minute appointment this week?" """)
    
    # Generate meta-prompt using few-shot prompting
    meta_prompt = f"""You are an AI sales representative for {customer_context.get('dealership_name', 'our automotive dealership')}. Generate an appointment booking message in the style of these examples:

{chr(10).join(examples)}

CUSTOMER CONTEXT:
- Name: {customer_context.get('first_name', '{first_name}')}
- Interested in: {customer_context.get('car_model', '{car_model}')}
- Customer Type: {customer_type_name}
- Conversation Stage: {stage_name}
- Urgency Level: {urgency_name}
- Budget: {customer_context.get('budget', '{budget}')}
- Timeline: {customer_context.get('timeline', '{timeline}')}

STRATEGY COMPONENTS TO USE:
{', '.join(selected_components)}

INSTRUCTIONS:
1. Follow the style and tone of the examples above
2. Adapt the approach to this specific customer type and situation
3. Use the recommended strategy components in your message
4. Match the conversation stage and urgency level
5. Keep it conversational and professional
6. End with a clear call-to-action for appointment booking

Generate a complete appointment booking message that follows the style of the examples and incorporates the specified strategy components."""

    return meta_prompt

def demo_meta_prompting():
    """Demonstrate the meta-prompting system."""
    print("🎯 META-PROMPTING SYSTEM DEMO")
    print("   (RL Agent generates prompts for other LLMs)")
    print("=" * 60)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        generator = AppointmentPromptGenerator()
        
        # Demo scenarios with different strategies and customer context
        scenarios = [
            {
                "name": "Cautious Customer - Early Stage",
                "customer_type": 0,  # cautious
                "conversation_stage": 0,  # early
                "urgency_level": 0,  # low
                "description": "Customer needs reassurance and information",
                "strategy_components": ["rapport_building", "needs_assessment", "value_proposition"],
                "customer_context": {
                    "first_name": "Sarah",
                    "car_model": "Honda Civic",
                    "dealership_name": "Metro Honda",
                    "budget": "$25,000",
                    "timeline": "within 2 months",
                    "key_benefit": "excellent fuel economy and reliability"
                }
            },
            {
                "name": "Ready Buyer - Closing Stage", 
                "customer_type": 2,  # ready_buyer
                "conversation_stage": 3,  # closing
                "urgency_level": 2,  # high
                "description": "Customer is ready to buy and needs immediate action",
                "strategy_components": ["urgency_creation", "appointment_booking", "incentive_offering"],
                "customer_context": {
                    "first_name": "Mike",
                    "car_model": "Ford F-150",
                    "dealership_name": "Premier Ford",
                    "budget": "$45,000",
                    "timeline": "next week",
                    "key_benefit": "powerful towing capacity and durability"
                }
            },
            {
                "name": "Price Shopper - Middle Stage",
                "customer_type": 1,  # price_shopper
                "conversation_stage": 1,  # middle
                "urgency_level": 1,  # medium
                "description": "Customer is comparing prices and needs incentives",
                "strategy_components": ["value_proposition", "incentive_offering", "social_proof"],
                "customer_context": {
                    "first_name": "Jennifer",
                    "car_model": "Toyota Camry",
                    "dealership_name": "City Toyota",
                    "budget": "$30,000",
                    "timeline": "this month",
                    "key_benefit": "long-term value and resale"
                }
            },
            {
                "name": "Skeptical Customer - Late Stage",
                "customer_type": 5,  # skeptical
                "conversation_stage": 2,  # late
                "urgency_level": 0,  # low
                "description": "Customer is hard to convince and needs proof",
                "strategy_components": ["objection_handling", "social_proof", "personalization"],
                "customer_context": {
                    "first_name": "Robert",
                    "car_model": "BMW 3 Series",
                    "dealership_name": "Luxury Motors",
                    "budget": "$40,000",
                    "timeline": "flexible",
                    "key_benefit": "premium performance and luxury"
                }
            }
        ]
        
        print(f"\n📝 Generating meta-prompts for {len(scenarios)} scenarios...")
        print("This demonstrates how the RL agent creates prompts for other LLMs:")
        print("1. RL agent analyzes customer context")
        print("2. RL agent selects optimal strategy components")
        print("3. System generates META-PROMPTS that instruct other LLMs")
        print("4. Meta-prompts can be used to generate actual sales messages")
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            print(f"Description: {scenario['description']}")
            
            # Calculate effectiveness for this strategy
            template = generator.generate_prompt_template(
                customer_type=scenario["customer_type"],
                conversation_stage=scenario["conversation_stage"],
                urgency_level=scenario["urgency_level"],
                selected_components=[list(generator.config["prompt_components"].keys()).index(comp) 
                                   for comp in scenario["strategy_components"]],
                customer_psychology=np.array([0.5, 0.3, 0.2, 0.4, 0.6, 0.3])  # Example psychology
            )
            
            # Generate meta-prompt
            meta_prompt = generate_meta_prompt_example(
                scenario["customer_type"],
                scenario["conversation_stage"], 
                scenario["urgency_level"],
                scenario["strategy_components"],
                scenario["customer_context"]
            )
            
            results.append({
                "scenario": scenario,
                "strategy": scenario["strategy_components"],
                "effectiveness": template.effectiveness_score,
                "meta_prompt": meta_prompt
            })
            
            print(f"✅ Strategy: {', '.join(scenario['strategy_components'])}")
            print(f"✅ Effectiveness: {template.effectiveness_score:.3f}")
        
        # Display results
        print(f"\n📊 META-PROMPTING RESULTS:")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\n--- Scenario {i+1}: {result['scenario']['name']} ---")
            print(f"RL Strategy: {', '.join(result['strategy'])}")
            print(f"Effectiveness: {result['effectiveness']:.3f}")
            print(f"\nGenerated META-PROMPT (for another LLM to use):")
            print("-" * 60)
            print(result['meta_prompt'])
            print("-" * 60)
        
        print(f"\n✅ SUCCESS! Generated {len(results)} meta-prompts")
        print("💡 These meta-prompts can now be used to instruct other LLMs")
        
        # Show example usage
        print(f"\n🎯 HOW TO USE THESE META-PROMPTS:")
        print("=" * 60)
        print("1. Take any meta-prompt from above")
        print("2. Use it as a system prompt for another LLM (GPT, Claude, etc.)")
        print("3. The other LLM will generate actual appointment booking messages")
        print("4. Each meta-prompt is optimized for specific customer types and situations")
        
        print(f"\n📝 EXAMPLE USAGE:")
        print("System Prompt: [Use the meta-prompt above]")
        print("User Message: 'Generate an appointment booking message for Sarah, who is interested in a Honda Civic'")
        print("LLM Response: [Professional appointment booking message]")
        
        print(f"\n🚀 KEY BENEFITS:")
        print("✅ RL agent learns optimal prompt strategies")
        print("✅ Meta-prompts adapt to different customer types")
        print("✅ Can be used with any LLM (GPT, Claude, etc.)")
        print("✅ Context-aware and situation-specific")
        print("✅ Professional and effective")
        
        # Show what the actual LLM would generate
        print(f"\n🤖 EXAMPLE: What an LLM would generate with these meta-prompts:")
        print("=" * 80)
        
        example_responses = [
            "Hi Sarah! I hope you're having a great day. I'm calling from Metro Honda about the Honda Civic you've been researching. I'd love to understand what you're looking for in your next vehicle. The Honda Civic offers excellent value with excellent fuel economy and reliability, and I'd love to help you find the perfect vehicle that meets your needs and budget. Would you be available for a 30-minute appointment when convenient?",
            
            "Mike, I know you're ready to make a decision on the Ford F-150, and I have some great news! We have limited availability on the Ford F-150 this week at Premier Ford. The Ford F-150 offers excellent value with powerful towing capacity and durability. I can get you a special financing rate of 2.9% and $500 cash back. Would you be available for a 1-hour appointment today or tomorrow?",
            
            "Jennifer, I understand you're comparing dealers for the Toyota Camry. The Toyota Camry offers excellent value with long-term value and resale at City Toyota. Many customers like you have been very satisfied with their Toyota Camry purchase. I can get you a special financing rate of 3.2% and extended warranty. Can we schedule a 45-minute appointment this week to discuss your specific needs?",
            
            "Robert, I understand your concerns about reliability and maintenance costs with the BMW 3 Series. Many customers like you have been very satisfied with their BMW 3 Series purchase at Luxury Motors. The BMW 3 Series offers excellent value with premium performance and luxury. I'd love to understand what you're looking for in your next vehicle and address any concerns you might have. Would you be available for a 1-hour appointment when convenient?"
        ]
        
        for i, (result, example) in enumerate(zip(results, example_responses)):
            print(f"\n--- Example {i+1}: {result['scenario']['name']} ---")
            print(f"Meta-prompt strategy: {', '.join(result['strategy'])}")
            print(f"Example LLM response:")
            print(f"'{example}'")
        
        print(f"\n🎉 CONCLUSION:")
        print("This demonstrates true meta-prompting - using RL to generate prompts")
        print("that instruct other LLMs to create effective appointment booking messages!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")

def main():
    """Main demo function."""
    try:
        demo_meta_prompting()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

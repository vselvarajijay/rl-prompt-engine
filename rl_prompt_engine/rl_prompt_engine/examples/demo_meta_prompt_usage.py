#!/usr/bin/env python3
"""
Demo script showing how to use the generated meta-prompts with different customers
"""

import json

def load_meta_prompts():
    """Load the generated meta-prompts."""
    with open("meta_prompts.json", "r") as f:
        return json.load(f)

def demonstrate_customer_usage():
    """Demonstrate how to use meta-prompts with different customers."""
    print("🎯 META-PROMPT USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Load meta-prompts
    meta_prompts = load_meta_prompts()
    
    # Different customer scenarios
    customers = [
        {
            "name": "Sarah Johnson",
            "model": "Honda Civic",
            "dealership": "Metro Honda",
            "budget": "$25,000",
            "timeline": "within 2 months",
            "key_benefit": "excellent fuel economy and reliability",
            "finance_rate": "2.9",
            "incentive": "$500 cash back",
            "appointment_duration": "30 minutes",
            "scenario": "Cautious Early"
        },
        {
            "name": "Mike Rodriguez",
            "model": "Ford F-150",
            "dealership": "Premier Ford",
            "budget": "$45,000",
            "timeline": "next week",
            "key_benefit": "powerful towing capacity and durability",
            "finance_rate": "3.2",
            "incentive": "extended warranty",
            "appointment_duration": "1 hour",
            "scenario": "Ready Buyer Closing"
        },
        {
            "name": "Jennifer Chen",
            "model": "Toyota Camry",
            "dealership": "City Toyota",
            "budget": "$30,000",
            "timeline": "this month",
            "key_benefit": "long-term value and resale",
            "finance_rate": "2.7",
            "incentive": "$750 cash back",
            "appointment_duration": "45 minutes",
            "scenario": "Price Shopper Middle"
        },
        {
            "name": "Robert Williams",
            "model": "BMW 3 Series",
            "dealership": "Luxury Motors",
            "budget": "$40,000",
            "timeline": "flexible",
            "key_benefit": "premium performance and luxury",
            "finance_rate": "4.1",
            "incentive": "free maintenance package",
            "appointment_duration": "1 hour",
            "scenario": "Skeptical Late"
        }
    ]
    
    print(f"📝 Demonstrating meta-prompt usage with {len(customers)} different customers:")
    print()
    
    for i, customer in enumerate(customers):
        print(f"--- Customer {i+1}: {customer['name']} ---")
        print(f"Scenario: {customer['scenario']}")
        print(f"Model: {customer['model']} | Budget: {customer['budget']}")
        
        # Find matching meta-prompt
        matching_meta_prompt = None
        for mp in meta_prompts:
            if mp["scenario_name"] == customer["scenario"]:
                matching_meta_prompt = mp
                break
        
        if matching_meta_prompt:
            # Personalize the meta-prompt
            personalized_prompt = matching_meta_prompt["meta_prompt"].format(
                First_Name=customer["name"],
                Model=customer["model"],
                Dealership_Name=customer["dealership"],
                Budget=customer["budget"],
                Timeline=customer["timeline"],
                Key_Benefit=customer["key_benefit"],
                Finance_Rate=customer["finance_rate"],
                Incentive=customer["incentive"],
                Appointment_Duration=customer["appointment_duration"]
            )
            
            print(f"✅ Strategy: {', '.join(matching_meta_prompt['strategy'])}")
            print(f"✅ Effectiveness: {matching_meta_prompt['effectiveness']:.3f}")
            print(f"✅ Personalized Meta-Prompt:")
            print("-" * 60)
            print(personalized_prompt[:300] + "...")
            print("-" * 60)
            
            # Show what the LLM would generate
            print(f"🤖 Example LLM Response:")
            if customer["scenario"] == "Cautious Early":
                response = f"Hi {customer['name']}! I hope you're having a great day. I'm calling from {customer['dealership']} about the {customer['model']} you've been researching. I'd love to help you find the perfect vehicle that meets your needs and budget. Would you be available for a {customer['appointment_duration']} appointment this week?"
            elif customer["scenario"] == "Ready Buyer Closing":
                response = f"Hi {customer['name']}! I know you're ready to make a decision on the {customer['model']}, and I have some great news! The {customer['model']} offers excellent value with {customer['key_benefit']}. I can get you a special financing rate of {customer['finance_rate']}% and {customer['incentive']}. Would you be available for a {customer['appointment_duration']} appointment today or tomorrow?"
            elif customer["scenario"] == "Price Shopper Middle":
                response = f"Hi {customer['name']}! I understand you're comparing dealers for the {customer['model']}. The {customer['model']} offers excellent value with {customer['key_benefit']} at {customer['dealership']}. I can get you a special financing rate of {customer['finance_rate']}% and {customer['incentive']}. Can we schedule a {customer['appointment_duration']} appointment this week?"
            else:  # Skeptical Late
                response = f"Hi {customer['name']}! I understand your concerns about reliability and maintenance costs with the {customer['model']}. Many customers like you have been very satisfied with their {customer['model']} purchase at {customer['dealership']}. I'd love to understand what you're looking for in your next vehicle and address any concerns you might have. Would you be available for a {customer['appointment_duration']} appointment when convenient?"
            
            print(f"'{response}'")
        else:
            print("❌ No matching meta-prompt found")
        
        print()
    
    print("🎯 KEY BENEFITS OF THIS SYSTEM:")
    print("=" * 60)
    print("✅ RL agent learns optimal strategies for different customer types")
    print("✅ Meta-prompts adapt to different scenarios and situations")
    print("✅ Variables allow personalization for any customer")
    print("✅ Few-shot examples show the desired style and tone")
    print("✅ Can be used with any LLM (GPT, Claude, etc.)")
    print("✅ Professional and effective appointment booking messages")
    print("✅ Scalable to thousands of customers and scenarios")
    
    print(f"\n🚀 USAGE INSTRUCTIONS:")
    print("=" * 60)
    print("1. Choose the appropriate meta-prompt for your customer scenario")
    print("2. Replace variables with actual customer data")
    print("3. Use as system prompt for any LLM")
    print("4. LLM will generate personalized appointment booking message")
    print("5. Repeat for different customers and scenarios")

def main():
    """Main demonstration function."""
    try:
        demonstrate_customer_usage()
    except FileNotFoundError:
        print("❌ meta_prompts.json not found. Please run end_to_end_meta_prompting.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

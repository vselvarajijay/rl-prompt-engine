#!/usr/bin/env python3
"""
Prompt Template Generator

This generates full prompt templates with parameters that can be used directly
to generate appointment booking messages for different customers.
"""

import json
import numpy as np
from appointment_prompt_env import AppointmentPromptEnv
from appointment_prompt_generator import AppointmentPromptGenerator
from stable_baselines3 import PPO

def generate_prompt_templates(model_path="ppo_meta_prompting"):
    """Generate full prompt templates with parameters."""
    print("🎯 GENERATING FULL PROMPT TEMPLATES")
    print("=" * 60)
    
    # Load trained model
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    except:
        print(f"❌ Model not found at {model_path}. Please train first.")
        return []
    
    env = AppointmentPromptEnv()
    generator = AppointmentPromptGenerator()
    
    # Customer scenarios
    scenarios = [
        {
            "name": "Cautious Early",
            "customer_type": 0,
            "conversation_stage": 0,
            "urgency_level": 0,
            "description": "Customer needs reassurance and information"
        },
        {
            "name": "Price Shopper Middle",
            "customer_type": 1,
            "conversation_stage": 1,
            "urgency_level": 1,
            "description": "Customer is comparing prices and needs incentives"
        },
        {
            "name": "Ready Buyer Closing",
            "customer_type": 2,
            "conversation_stage": 3,
            "urgency_level": 2,
            "description": "Customer is ready to buy and needs immediate action"
        },
        {
            "name": "Skeptical Late",
            "customer_type": 5,
            "conversation_stage": 2,
            "urgency_level": 0,
            "description": "Customer is hard to convince and needs proof"
        }
    ]
    
    prompt_templates = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Template {i+1}: {scenario['name']} ---")
        
        # Generate strategy using RL agent
        obs, _ = env.reset()
        obs["customer_type"][0] = scenario["customer_type"]
        obs["conversation_stage"][0] = scenario["conversation_stage"]
        obs["urgency_level"][0] = scenario["urgency_level"]
        
        selected_components = []
        step = 0
        max_steps = 6
        
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0]) if hasattr(action, '__len__') else int(action)
            finish = action % 2
            position = (action // 2) % 6
            component_idx = (action // (2 * 6)) % 10
            
            if finish == 1:
                break
            
            if component_idx not in selected_components and position < 6:
                selected_components.append(component_idx)
                obs["prompt_so_far"][position] = 1.0
                obs["turn"][0] = step + 1
            
            step += 1
        
        # Get component names
        component_names = list(generator.config["prompt_components"].keys())
        selected_component_names = [component_names[i] for i in selected_components]
        
        # Calculate effectiveness
        template = generator.generate_prompt_template(
            customer_type=scenario["customer_type"],
            conversation_stage=scenario["conversation_stage"],
            urgency_level=scenario["urgency_level"],
            selected_components=selected_components,
            customer_psychology=obs["customer_psychology"]
        )
        
        # Generate full prompt template
        prompt_template = generate_full_prompt_template(
            scenario, selected_component_names, template.effectiveness_score
        )
        
        prompt_templates.append({
            "scenario": scenario,
            "strategy": selected_component_names,
            "effectiveness": template.effectiveness_score,
            "template": prompt_template
        })
        
        print(f"✅ Strategy: {', '.join(selected_component_names)}")
        print(f"✅ Effectiveness: {template.effectiveness_score:.3f}")
        print(f"✅ Template generated with parameters")
    
    return prompt_templates

def generate_full_prompt_template(scenario, strategy_components, effectiveness):
    """Generate a full prompt template with parameters."""
    
    # Determine tone and approach based on customer type
    customer_type = scenario["customer_type"]
    if customer_type == 0:  # cautious
        tone = "warm, reassuring, and patient"
        approach = "build trust gradually, provide detailed information"
    elif customer_type == 1:  # price_shopper
        tone = "confident, value-focused, and competitive"
        approach = "emphasize value, offer incentives, highlight savings"
    elif customer_type == 2:  # ready_buyer
        tone = "direct, efficient, and action-oriented"
        approach = "move quickly to closing, emphasize availability"
    elif customer_type == 5:  # skeptical
        tone = "respectful, evidence-based, and transparent"
        approach = "address concerns directly, provide proof and testimonials"
    else:
        tone = "professional, friendly, and helpful"
        approach = "adapt to customer needs, focus on benefits"
    
    # Determine urgency level
    urgency_level = scenario["urgency_level"]
    if urgency_level == 2:  # high
        tone += " with a sense of urgency"
        time_reference = "today or tomorrow"
    elif urgency_level == 1:  # medium
        time_reference = "this week"
    else:  # low
        time_reference = "when convenient"
    
    # Build template based on strategy components
    template_parts = []
    
    if "rapport_building" in strategy_components:
        template_parts.append("Hi {first_name}! I hope you're having a great day.")
    
    if "needs_assessment" in strategy_components:
        template_parts.append("I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car?")
    
    if "value_proposition" in strategy_components:
        template_parts.append("The {car_model} offers excellent value with {key_benefit}.")
    
    if "urgency_creation" in strategy_components:
        template_parts.append("We have limited availability on the {car_model} this week.")
    
    if "incentive_offering" in strategy_components:
        template_parts.append("I can get you a special financing rate of {finance_rate}% and {incentive}.")
    
    if "social_proof" in strategy_components:
        template_parts.append("Many customers like you have been very satisfied with their {car_model} purchase.")
    
    if "objection_handling" in strategy_components:
        template_parts.append("I understand your concerns about {concern} with the {car_model}.")
    
    if "appointment_booking" in strategy_components:
        template_parts.append("Would you be available for a {appointment_duration} appointment {time_reference}?")
    
    # Create the full prompt template
    template = f"""You are an AI sales representative for {scenario['name']} customers. Generate an appointment booking message with the following specifications:

CUSTOMER PROFILE:
- Customer Type: {scenario['name']}
- Conversation Stage: {scenario['conversation_stage']}
- Urgency Level: {scenario['urgency_level']}
- Description: {scenario['description']}

TONE AND APPROACH:
- Tone: {tone}
- Approach: {approach}
- Time Reference: {time_reference}

MESSAGE TEMPLATE:
{chr(10).join(template_parts)}

PARAMETERS TO FILL:
- first_name: Customer's first name
- car_model: Specific car model they're interested in
- dealership_name: Name of your dealership
- budget: Customer's budget range
- timeline: When they need the car
- key_benefit: Main benefit of the car (fuel economy, reliability, etc.)
- finance_rate: Special financing rate (e.g., 2.9)
- incentive: Special offer (e.g., $500 cash back, extended warranty)
- appointment_duration: Appointment length (e.g., 30 minutes, 1 hour)
- concern: Specific concern to address (e.g., reliability, maintenance costs)
- time_reference: When to schedule (e.g., today, this week, when convenient)

INSTRUCTIONS:
1. Fill in all parameters with appropriate values
2. Use the {tone} tone throughout
3. Incorporate all template parts in a natural flow
4. Keep the message conversational and professional
5. End with a clear call-to-action for appointment booking
6. Make it sound like a real sales conversation

Generate a complete appointment booking message that follows this template and incorporates all specified elements."""

    return template

def demonstrate_template_usage(prompt_templates):
    """Demonstrate how to use the prompt templates."""
    print("\n🎯 DEMONSTRATING TEMPLATE USAGE")
    print("=" * 60)
    
    # Example customer data
    customers = [
        {
            "first_name": "Sarah",
            "car_model": "Honda Civic",
            "dealership_name": "Metro Honda",
            "budget": "$25,000",
            "timeline": "within 2 months",
            "key_benefit": "excellent fuel economy and reliability",
            "finance_rate": "2.9",
            "incentive": "$500 cash back",
            "appointment_duration": "30 minutes",
            "concern": "reliability and maintenance costs",
            "time_reference": "this week"
        },
        {
            "first_name": "Mike",
            "car_model": "Ford F-150",
            "dealership_name": "Premier Ford",
            "budget": "$45,000",
            "timeline": "next week",
            "key_benefit": "powerful towing capacity and durability",
            "finance_rate": "3.2",
            "incentive": "extended warranty",
            "appointment_duration": "1 hour",
            "concern": "fuel efficiency",
            "time_reference": "today or tomorrow"
        }
    ]
    
    for i, customer in enumerate(customers):
        print(f"\n--- Customer {i+1}: {customer['first_name']} ---")
        
        # Find matching template
        matching_template = None
        for template in prompt_templates:
            if template["scenario"]["name"] == "Cautious Early" if i == 0 else "Ready Buyer Closing":
                matching_template = template
                break
        
        if matching_template:
            # Fill in the template
            filled_template = matching_template["template"].format(**customer)
            
            print(f"✅ Using template: {matching_template['scenario']['name']}")
            print(f"✅ Strategy: {', '.join(matching_template['strategy'])}")
            print(f"✅ Effectiveness: {matching_template['effectiveness']:.3f}")
            print(f"\n📋 FILLED TEMPLATE:")
            print("-" * 60)
            print(filled_template)
            print("-" * 60)
            
            # Show what the LLM would generate
            print(f"\n🤖 EXAMPLE LLM RESPONSE:")
            if i == 0:  # Cautious Early
                response = f"Hi {customer['first_name']}! I hope you're having a great day. I'm calling from {customer['dealership_name']} about the {customer['car_model']} you've been researching. I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car? The {customer['car_model']} offers excellent value with {customer['key_benefit']}. Would you be available for a {customer['appointment_duration']} appointment {customer['time_reference']}?"
            else:  # Ready Buyer Closing
                response = f"Hi {customer['first_name']}! I know you're ready to make a decision on the {customer['car_model']}, and I have some great news! The {customer['car_model']} offers excellent value with {customer['key_benefit']}. We have limited availability on the {customer['car_model']} this week. I can get you a special financing rate of {customer['finance_rate']}% and {customer['incentive']}. Would you be available for a {customer['appointment_duration']} appointment {customer['time_reference']}?"
            
            print(f"'{response}'")
        else:
            print("❌ No matching template found")
        
        print()

def save_templates(prompt_templates):
    """Save prompt templates to file."""
    print("\n💾 SAVING PROMPT TEMPLATES")
    print("=" * 60)
    
    # Save templates to JSON
    templates_data = []
    for template in prompt_templates:
        templates_data.append({
            "scenario_name": template["scenario"]["name"],
            "customer_type": template["scenario"]["customer_type"],
            "conversation_stage": template["scenario"]["conversation_stage"],
            "urgency_level": template["scenario"]["urgency_level"],
            "strategy": template["strategy"],
            "effectiveness": template["effectiveness"],
            "template": template["template"]
        })
    
    with open("prompt_templates.json", "w") as f:
        json.dump(templates_data, f, indent=2)
    
    print(f"✅ Saved {len(templates_data)} prompt templates to prompt_templates.json")
    
    # Also save as a simple text file for easy reading
    with open("prompt_templates.txt", "w") as f:
        for i, template in enumerate(templates_data):
            f.write(f"=== TEMPLATE {i+1}: {template['scenario_name']} ===\n")
            f.write(f"Strategy: {', '.join(template['strategy'])}\n")
            f.write(f"Effectiveness: {template['effectiveness']:.3f}\n")
            f.write(f"Template:\n{template['template']}\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"✅ Also saved to prompt_templates.txt for easy reading")

def main():
    """Main function to generate prompt templates."""
    print("🚀 PROMPT TEMPLATE GENERATOR")
    print("=" * 60)
    print("This will generate full prompt templates with parameters")
    print("that can be used directly to generate appointment booking messages.")
    print("=" * 60)
    
    try:
        # Generate prompt templates
        prompt_templates = generate_prompt_templates()
        
        if prompt_templates:
            # Demonstrate usage
            demonstrate_template_usage(prompt_templates)
            
            # Save templates
            save_templates(prompt_templates)
            
            print(f"\n🎉 PROMPT TEMPLATE GENERATION COMPLETED!")
            print("=" * 60)
            print("✅ Generated full prompt templates with parameters")
            print("✅ Demonstrated usage with real customer data")
            print("✅ Saved templates to files")
            print("\n🚀 You can now use these templates with any LLM!")
        else:
            print("❌ No templates generated. Please check your model.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

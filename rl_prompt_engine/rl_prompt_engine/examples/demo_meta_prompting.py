#!/usr/bin/env python3
"""
Demo script for Meta-Prompting System (RL Agent generates prompts for other LLMs)

This demonstrates the correct approach:
1. RL agent determines optimal prompt strategy
2. System generates META-PROMPTS that instruct other LLMs
3. Meta-prompts are used to generate actual appointment booking messages
"""

from appointment_prompt_env import AppointmentPromptEnv
from appointment_prompt_generator import AppointmentPromptGenerator
from stable_baselines3 import PPO
import numpy as np

def generate_meta_prompt_example(customer_type, conversation_stage, urgency_level, selected_components):
    """Generate a meta-prompt example based on RL strategy."""
    
    customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
    conversation_stages = ["early", "middle", "late", "closing"]
    urgency_levels = ["low", "medium", "high"]
    
    customer_type_name = customer_types[customer_type]
    stage_name = conversation_stages[conversation_stage]
    urgency_name = urgency_levels[urgency_level]
    
    # Generate meta-prompt based on strategy
    meta_prompt = f"""You are an AI sales representative for an automotive dealership. Generate an appointment booking message with the following specifications:

CUSTOMER PROFILE:
- Customer Type: {customer_type_name}
- Conversation Stage: {stage_name}
- Urgency Level: {urgency_name}

STRATEGY COMPONENTS TO USE:
{', '.join(selected_components)}

INSTRUCTIONS:
1. Adapt your approach to this specific customer type and situation
2. Use the recommended strategy components in your message
3. Match the conversation stage and urgency level
4. Generate a professional, natural-sounding appointment booking message
5. Keep it concise but effective

TONE AND STYLE:
- Professional but approachable
- Customer-focused and helpful
- Appropriate for the customer type and situation
- Clear call-to-action for appointment booking

Generate a single appointment booking message that incorporates these elements."""

    return meta_prompt

def demo_meta_prompting():
    """Demonstrate the meta-prompting system."""
    print("🎯 META-PROMPTING SYSTEM DEMO")
    print("   (RL Agent generates prompts for other LLMs)")
    print("=" * 60)
    
    try:
        # Initialize components
        print("🔧 Initializing RL components...")
        env = AppointmentPromptEnv()
        generator = AppointmentPromptGenerator()
        
        # Load trained model
        try:
            model = PPO.load("ppo_appointment_prompts")
            print("✅ RL model loaded successfully")
        except:
            print("❌ RL model not found. Please train first with: python train_appointment_prompts.py")
            return
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Cautious Customer - Early Stage",
                "customer_type": 0,  # cautious
                "conversation_stage": 0,  # early
                "urgency_level": 0,  # low
                "description": "Customer needs reassurance and information"
            },
            {
                "name": "Ready Buyer - Closing Stage", 
                "customer_type": 2,  # ready_buyer
                "conversation_stage": 3,  # closing
                "urgency_level": 2,  # high
                "description": "Customer is ready to buy and needs immediate action"
            },
            {
                "name": "Price Shopper - Middle Stage",
                "customer_type": 1,  # price_shopper
                "conversation_stage": 1,  # middle
                "urgency_level": 1,  # medium
                "description": "Customer is comparing prices and needs incentives"
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
            
            # Generate meta-prompt
            meta_prompt = generate_meta_prompt_example(
                scenario["customer_type"],
                scenario["conversation_stage"], 
                scenario["urgency_level"],
                selected_component_names
            )
            
            results.append({
                "scenario": scenario,
                "strategy": selected_component_names,
                "effectiveness": template.effectiveness_score,
                "meta_prompt": meta_prompt
            })
            
            print(f"✅ Strategy: {', '.join(selected_component_names)}")
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

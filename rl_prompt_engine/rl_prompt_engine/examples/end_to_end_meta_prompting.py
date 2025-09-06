#!/usr/bin/env python3
"""
Complete End-to-End Meta-Prompting System

This script provides the full pipeline:
1. Train RL model
2. Evaluate and generate prompt templates
3. Create meta-prompts with variables for different customers
"""

import os
import json
import numpy as np
from appointment_prompt_env import AppointmentPromptEnv
from appointment_prompt_generator import AppointmentPromptGenerator, AppointmentPromptDatabase
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import time

def train_rl_model(total_timesteps=50000):
    """Train the RL model for meta-prompting."""
    print("🚀 TRAINING RL MODEL FOR META-PROMPTING")
    print("=" * 60)
    
    # Create environment
    env = AppointmentPromptEnv()
    vec_env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: AppointmentPromptEnv()])
    
    # Initialize PPO model
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    print(f"📚 Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Save the model
    model_path = "ppo_meta_prompting"
    model.save(model_path)
    print(f"💾 Model saved as {model_path}")
    
    return model, model_path

def evaluate_and_generate_templates(model, num_episodes=100):
    """Evaluate the trained model and generate prompt templates."""
    print("\n📊 EVALUATING MODEL AND GENERATING TEMPLATES")
    print("=" * 60)
    
    env = AppointmentPromptEnv()
    generator = AppointmentPromptGenerator()
    database = AppointmentPromptDatabase()
    
    # Customer scenarios to test
    scenarios = [
        {"customer_type": 0, "conversation_stage": 0, "urgency_level": 0, "name": "Cautious Early"},
        {"customer_type": 1, "conversation_stage": 1, "urgency_level": 1, "name": "Price Shopper Middle"},
        {"customer_type": 2, "conversation_stage": 3, "urgency_level": 2, "name": "Ready Buyer Closing"},
        {"customer_type": 5, "conversation_stage": 2, "urgency_level": 0, "name": "Skeptical Late"},
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        
        # Reset environment with specific scenario
        obs, _ = env.reset()
        obs["customer_type"][0] = scenario["customer_type"]
        obs["conversation_stage"][0] = scenario["conversation_stage"]
        obs["urgency_level"][0] = scenario["urgency_level"]
        
        # Generate strategy using RL agent
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
        
        # Store in database
        database.add_template(template)
        
        results.append({
            "scenario": scenario,
            "strategy": selected_component_names,
            "effectiveness": template.effectiveness_score,
            "template": template
        })
        
        print(f"✅ Strategy: {', '.join(selected_component_names)}")
        print(f"✅ Effectiveness: {template.effectiveness_score:.3f}")
    
    print(f"\n📈 Generated {len(results)} prompt templates")
    return results, database

def generate_meta_prompts_with_variables(results, database):
    """Generate meta-prompts with variables for different customers."""
    print("\n🎯 GENERATING META-PROMPTS WITH VARIABLES")
    print("=" * 60)
    
    # Customer contexts with variables
    customer_contexts = [
        {
            "first_name": "{First_Name}",
            "car_model": "{Model}",
            "dealership_name": "{Dealership_Name}",
            "budget": "{Budget}",
            "timeline": "{Timeline}",
            "key_benefit": "{Key_Benefit}",
            "interest_rate": "{Finance_Rate}",
            "incentive": "{Incentive}",
            "time_duration": "{Appointment_Duration}"
        }
    ]
    
    meta_prompts = []
    
    for i, result in enumerate(results):
        scenario = result["scenario"]
        strategy = result["strategy"]
        effectiveness = result["effectiveness"]
        
        # Generate few-shot examples based on strategy
        examples = generate_few_shot_examples(strategy, customer_contexts[0])
        
        # Create meta-prompt with variables
        meta_prompt = f"""You are an AI sales representative for {{Dealership_Name}}. Generate an appointment booking message in the style of these examples:

{chr(10).join(examples)}

CUSTOMER CONTEXT VARIABLES:
- Name: {{First_Name}}
- Interested in: {{Model}}
- Customer Type: {scenario['name']}
- Budget: {{Budget}}
- Timeline: {{Timeline}}
- Key Benefit: {{Key_Benefit}}
- Finance Rate: {{Finance_Rate}}%
- Incentive: {{Incentive}}
- Appointment Duration: {{Appointment_Duration}}

STRATEGY COMPONENTS TO USE:
{', '.join(strategy)}

INSTRUCTIONS:
1. Follow the style and tone of the examples above
2. Use the provided variables in your message
3. Adapt the approach to this specific customer type and situation
4. Use the recommended strategy components in your message
5. Keep it conversational and professional
6. End with a clear call-to-action for appointment booking

Generate a complete appointment booking message that follows the style of the examples and incorporates the specified strategy components using the provided variables."""

        meta_prompts.append({
            "scenario": scenario,
            "strategy": strategy,
            "effectiveness": effectiveness,
            "meta_prompt": meta_prompt
        })
        
        print(f"\n--- Meta-Prompt {i+1}: {scenario['name']} ---")
        print(f"Strategy: {', '.join(strategy)}")
        print(f"Effectiveness: {effectiveness:.3f}")
        print(f"Variables: First_Name, Model, Dealership_Name, Budget, Timeline, Key_Benefit, Finance_Rate, Incentive, Appointment_Duration")
    
    return meta_prompts

def generate_few_shot_examples(strategy, context):
    """Generate few-shot examples based on strategy components."""
    examples = []
    
    if "rapport_building" in strategy and "needs_assessment" in strategy:
        examples.append(f"""Example 1: "Hi {context['first_name']}! I hope you're having a great day. I'm calling from {context['dealership_name']} about the {context['car_model']} you've been researching. I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car?" """)
    
    if "value_proposition" in strategy and "urgency_creation" in strategy:
        examples.append(f"""Example 2: "The {context['car_model']} offers excellent value with {context['key_benefit']}, and we have limited availability this week. I can get you a special financing rate of {context['interest_rate']}% and {context['incentive']}. Would you be available for a {context['time_duration']} appointment today or tomorrow?" """)
    
    if "social_proof" in strategy and "incentive_offering" in strategy:
        examples.append(f"""Example 3: "Many customers like you have been very satisfied with their {context['car_model']} purchase at {context['dealership_name']}. The {context['car_model']} offers excellent value with {context['key_benefit']}. I can get you a special financing rate of {context['interest_rate']}% and {context['incentive']}. Can we schedule a {context['time_duration']} appointment this week?" """)
    
    if "objection_handling" in strategy and "personalization" in strategy:
        examples.append(f"""Example 4: "I understand your concerns about reliability and maintenance costs with the {context['car_model']}. Many customers like you have been very satisfied with their {context['car_model']} purchase at {context['dealership_name']}. I'd love to understand what you're looking for in your next vehicle and address any concerns you might have. Would you be available for a {context['time_duration']} appointment when convenient?" """)
    
    # If no examples match, create a generic one
    if not examples:
        examples.append(f"""Example 1: "Hi {context['first_name']}! I hope you're having a great day. I'm calling from {context['dealership_name']} about the {context['car_model']} you've been researching. I'd love to help you find the perfect vehicle that meets your needs and budget. Would you be available for a {context['time_duration']} appointment this week?" """)
    
    return examples

def demonstrate_usage(meta_prompts):
    """Demonstrate how to use the generated meta-prompts."""
    print("\n🎯 DEMONSTRATING META-PROMPT USAGE")
    print("=" * 60)
    
    # Example customer data
    customer_data = {
        "First_Name": "Sarah",
        "Model": "Honda Civic",
        "Dealership_Name": "Metro Honda",
        "Budget": "$25,000",
        "Timeline": "within 2 months",
        "Key_Benefit": "excellent fuel economy and reliability",
        "Finance_Rate": "2.9",
        "Incentive": "$500 cash back",
        "Appointment_Duration": "30 minutes"
    }
    
    print("📝 Example Customer Data:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
    
    print(f"\n🤖 Example Usage:")
    print("1. Take any meta-prompt from above")
    print("2. Replace variables with actual customer data")
    print("3. Use as system prompt for any LLM")
    print("4. LLM will generate personalized appointment booking message")
    
    # Show example with actual data
    if meta_prompts:
        example_meta_prompt = meta_prompts[0]["meta_prompt"]
        personalized_prompt = example_meta_prompt.format(**customer_data)
        
        print(f"\n📋 Example Personalized Meta-Prompt:")
        print("-" * 60)
        print(personalized_prompt[:500] + "...")
        print("-" * 60)
        
        print(f"\n💡 The LLM would then generate a message like:")
        print("'Hi Sarah! I hope you're having a great day. I'm calling from Metro Honda about the Honda Civic you've been researching. I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car? The Honda Civic offers excellent value with excellent fuel economy and reliability. Would you be available for a 30-minute appointment this week?'")

def save_results(meta_prompts, database):
    """Save results to files."""
    print("\n💾 SAVING RESULTS")
    print("=" * 60)
    
    # Save meta-prompts to JSON
    meta_prompts_data = []
    for mp in meta_prompts:
        meta_prompts_data.append({
            "scenario_name": mp["scenario"]["name"],
            "customer_type": mp["scenario"]["customer_type"],
            "conversation_stage": mp["scenario"]["conversation_stage"],
            "urgency_level": mp["scenario"]["urgency_level"],
            "strategy": mp["strategy"],
            "effectiveness": mp["effectiveness"],
            "meta_prompt": mp["meta_prompt"]
        })
    
    with open("meta_prompts.json", "w") as f:
        json.dump(meta_prompts_data, f, indent=2)
    
    print(f"✅ Saved {len(meta_prompts_data)} meta-prompts to meta_prompts.json")
    
    # Save database statistics
    stats = database.get_statistics()
    with open("database_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved database statistics to database_stats.json")
    print(f"📊 Database contains {stats['total_templates']} templates")
    print(f"📊 Average effectiveness: {stats['avg_effectiveness']:.3f}")

def main():
    """Main end-to-end pipeline."""
    print("🚀 COMPLETE END-TO-END META-PROMPTING SYSTEM")
    print("=" * 80)
    print("This system will:")
    print("1. Train RL model for meta-prompting")
    print("2. Evaluate and generate prompt templates")
    print("3. Create meta-prompts with variables for different customers")
    print("4. Demonstrate usage with real customer data")
    print("=" * 80)
    
    try:
        # Step 1: Train RL model
        model, model_path = train_rl_model(total_timesteps=10000)  # Reduced for demo
        
        # Step 2: Evaluate and generate templates
        results, database = evaluate_and_generate_templates(model)
        
        # Step 3: Generate meta-prompts with variables
        meta_prompts = generate_meta_prompts_with_variables(results, database)
        
        # Step 4: Demonstrate usage
        demonstrate_usage(meta_prompts)
        
        # Step 5: Save results
        save_results(meta_prompts, database)
        
        print(f"\n🎉 END-TO-END SYSTEM COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ RL model trained and saved")
        print("✅ Prompt templates generated and evaluated")
        print("✅ Meta-prompts with variables created")
        print("✅ Usage demonstrated with real customer data")
        print("✅ Results saved to files")
        print("\n🚀 You can now use these meta-prompts with any LLM!")
        
    except Exception as e:
        print(f"❌ Error in end-to-end pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

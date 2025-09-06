#!/usr/bin/env python3
"""
Demo script for the Appointment Booking Meta-Prompting System

This script demonstrates how the system works by:
1. Training a simple model
2. Generating prompts for different scenarios
3. Showing the effectiveness of different approaches
"""

import numpy as np
from appointment_prompt_env import AppointmentPromptEnv
from appointment_prompt_generator import AppointmentPromptGenerator, AppointmentPromptDatabase
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def demo_system():
    """Demonstrate the appointment booking meta-prompting system."""
    print("🎯 APPOINTMENT BOOKING META-PROMPTING SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows how AI learns to create optimal prompts for booking appointments.")
    print()
    
    # Step 1: Create and train a simple model
    print("🚀 Step 1: Training the AI agent...")
    print("(This may take a few minutes)")
    
    # Create environment
    env = make_vec_env(lambda: AppointmentPromptEnv(), n_envs=4)
    
    # Train a simple model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0
    )
    
    # Train for a short time for demo
    model.learn(total_timesteps=50000)
    print("✅ Training complete!")
    print()
    
    # Step 2: Initialize generator and database
    print("🔧 Step 2: Setting up prompt generator...")
    generator = AppointmentPromptGenerator()
    database = AppointmentPromptDatabase()
    print("✅ Generator ready!")
    print()
    
    # Step 3: Generate prompts for different scenarios
    print("📝 Step 3: Generating prompts for different customer types...")
    
    customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
    conversation_stages = ["early", "middle", "late", "closing"]
    urgency_levels = ["low", "medium", "high"]
    
    # Generate prompts for a few scenarios
    scenarios = [
        (0, 0, 0),  # cautious, early, low urgency
        (2, 3, 2),  # ready_buyer, closing, high urgency
        (1, 1, 1),  # price_shopper, middle, medium urgency
        (5, 2, 0),  # skeptical, late, low urgency
    ]
    
    generated_templates = []
    
    for i, (customer_type, conversation_stage, urgency_level) in enumerate(scenarios):
        print(f"\n--- Scenario {i+1}: {customer_types[customer_type]} customer, {conversation_stages[conversation_stage]} stage, {urgency_levels[urgency_level]} urgency ---")
        
        # Create environment and set context
        env = AppointmentPromptEnv()
        obs, _ = env.reset()
        obs["customer_type"][0] = customer_type
        obs["conversation_stage"][0] = conversation_stage
        obs["urgency_level"][0] = urgency_level
        
        # Generate prompt using AI
        selected_components = []
        step = 0
        max_steps = 6
        
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            # Decode flattened action
            action = int(action[0]) if hasattr(action, '__len__') else int(action)
            finish = action % 2
            position = (action // 2) % 6  # MAX_PROMPT_LENGTH
            component_idx = (action // (2 * 6)) % 10  # N_PROMPT_COMPONENTS
            
            if finish == 1:
                break
            
            if component_idx not in selected_components and position < 6:
                selected_components.append(component_idx)
                obs["prompt_so_far"][position] = 1.0
                obs["turn"][0] = step + 1
            
            step += 1
        
        # Generate actual prompt template
        template = generator.generate_prompt_template(
            customer_type=customer_type,
            conversation_stage=conversation_stage,
            urgency_level=urgency_level,
            selected_components=selected_components,
            customer_psychology=obs["customer_psychology"]
        )
        
        generated_templates.append(template)
        
        # Display results
        print(f"Components selected: {', '.join(template.components)}")
        print(f"Effectiveness score: {template.effectiveness_score:.3f}")
        print(f"Prompt preview: {template.template[:150]}...")
        
        # Save to database
        database.add_template(template)
    
    print("\n✅ Generated prompts for all scenarios!")
    print()
    
    # Step 4: Show database statistics
    print("📊 Step 4: Database statistics...")
    stats = database.get_statistics()
    print(f"Total templates: {stats['total_templates']}")
    print(f"Average effectiveness: {stats['avg_effectiveness']:.3f}")
    print(f"Max effectiveness: {stats['max_effectiveness']:.3f}")
    print()
    
    # Step 5: Show best template
    print("🏆 Step 5: Best performing template...")
    all_templates = list(database.templates.values())
    best_template = max(all_templates, key=lambda x: x.effectiveness_score)
    
    print(f"Customer type: {best_template.customer_type}")
    print(f"Conversation stage: {best_template.conversation_stage}")
    print(f"Urgency level: {best_template.urgency_level}")
    print(f"Effectiveness: {best_template.effectiveness_score:.3f}")
    print(f"Components: {', '.join(best_template.components)}")
    print("\nFull prompt:")
    print("-" * 60)
    print(best_template.template)
    print("-" * 60)
    print()
    
    # Step 6: Show how to use the system
    print("💡 Step 6: How to use this system...")
    print("""
This system can be used to:

1. 🎭 Generate prompts for any customer type and situation
2. 📊 Test prompt effectiveness across different scenarios  
3. 🔍 Search existing prompt templates
4. 📈 View system performance and statistics
5. 🧪 Train new models with different data

To use the full system:
- Run: python appointment_prompt_interface.py
- Or: python train_appointment_prompts.py (to train)
- Or: python eval_appointment_prompts.py (to evaluate)

The AI learns which prompt components work best for different:
- Customer types (cautious, price_shopper, ready_buyer, etc.)
- Conversation stages (early, middle, late, closing)
- Urgency levels (low, medium, high)
- Customer psychology states
""")
    
    print("🎉 Demo complete! The system is ready to generate appointment booking prompts.")

def main():
    """Main demo function."""
    try:
        demo_system()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure all dependencies are installed and the environment is set up correctly.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for the Appointment Booking Meta-Prompting RL Agent

This script trains a PPO agent to learn optimal prompt construction strategies
for generating appointment booking prompts that work with different customer types
and conversation contexts.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from .core.appointment_prompt_env import AppointmentPromptEnv
import json

def create_appointment_prompt_config():
    """Create the appointment prompt configuration file."""
    config = {
        "prompt_components": {
            "rapport_building": {
                "description": "Build rapport and establish connection",
                "effectiveness": {
                    "cautious": 0.9, "price_shopper": 0.6, "ready_buyer": 0.7,
                    "research_buyer": 0.8, "impulse_buyer": 0.5, "skeptical": 0.8
                },
                "stage_effectiveness": {"early": 0.9, "middle": 0.7, "late": 0.4, "closing": 0.2},
                "compatibility": ["needs_assessment", "value_proposition"]
            },
            "needs_assessment": {
                "description": "Assess customer needs and preferences",
                "effectiveness": {
                    "cautious": 0.8, "price_shopper": 0.7, "ready_buyer": 0.6,
                    "research_buyer": 0.9, "impulse_buyer": 0.4, "skeptical": 0.7
                },
                "stage_effectiveness": {"early": 0.8, "middle": 0.9, "late": 0.6, "closing": 0.3},
                "compatibility": ["rapport_building", "value_proposition"]
            },
            "value_proposition": {
                "description": "Present value and benefits",
                "effectiveness": {
                    "cautious": 0.7, "price_shopper": 0.9, "ready_buyer": 0.8,
                    "research_buyer": 0.8, "impulse_buyer": 0.6, "skeptical": 0.7
                },
                "stage_effectiveness": {"early": 0.5, "middle": 0.8, "late": 0.9, "closing": 0.7},
                "compatibility": ["needs_assessment", "objection_handling"]
            },
            "objection_handling": {
                "description": "Address concerns and objections",
                "effectiveness": {
                    "cautious": 0.8, "price_shopper": 0.9, "ready_buyer": 0.5,
                    "research_buyer": 0.8, "impulse_buyer": 0.4, "skeptical": 0.9
                },
                "stage_effectiveness": {"early": 0.4, "middle": 0.7, "late": 0.8, "closing": 0.9},
                "compatibility": ["value_proposition", "urgency_creation"]
            },
            "urgency_creation": {
                "description": "Create urgency and time pressure",
                "effectiveness": {
                    "cautious": 0.4, "price_shopper": 0.8, "ready_buyer": 0.7,
                    "research_buyer": 0.3, "impulse_buyer": 0.9, "skeptical": 0.5
                },
                "stage_effectiveness": {"early": 0.3, "middle": 0.6, "late": 0.8, "closing": 0.9},
                "compatibility": ["objection_handling", "social_proof"]
            },
            "social_proof": {
                "description": "Provide social proof and testimonials",
                "effectiveness": {
                    "cautious": 0.8, "price_shopper": 0.6, "ready_buyer": 0.5,
                    "research_buyer": 0.9, "impulse_buyer": 0.4, "skeptical": 0.8
                },
                "stage_effectiveness": {"early": 0.6, "middle": 0.8, "late": 0.7, "closing": 0.5},
                "compatibility": ["urgency_creation", "incentive_offering"]
            },
            "incentive_offering": {
                "description": "Offer incentives and special deals",
                "effectiveness": {
                    "cautious": 0.6, "price_shopper": 0.9, "ready_buyer": 0.7,
                    "research_buyer": 0.5, "impulse_buyer": 0.8, "skeptical": 0.6
                },
                "stage_effectiveness": {"early": 0.4, "middle": 0.6, "late": 0.8, "closing": 0.9},
                "compatibility": ["social_proof", "appointment_booking"]
            },
            "appointment_booking": {
                "description": "Directly request appointment booking",
                "effectiveness": {
                    "cautious": 0.5, "price_shopper": 0.7, "ready_buyer": 0.9,
                    "research_buyer": 0.4, "impulse_buyer": 0.8, "skeptical": 0.6
                },
                "stage_effectiveness": {"early": 0.2, "middle": 0.4, "late": 0.8, "closing": 0.9},
                "compatibility": ["incentive_offering", "follow_up"]
            },
            "follow_up": {
                "description": "Set up follow-up and next steps",
                "effectiveness": {
                    "cautious": 0.7, "price_shopper": 0.6, "ready_buyer": 0.5,
                    "research_buyer": 0.8, "impulse_buyer": 0.4, "skeptical": 0.7
                },
                "stage_effectiveness": {"early": 0.3, "middle": 0.5, "late": 0.7, "closing": 0.8},
                "compatibility": ["appointment_booking"]
            },
            "personalization": {
                "description": "Personalize the approach based on customer data",
                "effectiveness": {
                    "cautious": 0.8, "price_shopper": 0.7, "ready_buyer": 0.6,
                    "research_buyer": 0.9, "impulse_buyer": 0.5, "skeptical": 0.8
                },
                "stage_effectiveness": {"early": 0.7, "middle": 0.8, "late": 0.6, "closing": 0.4},
                "compatibility": ["rapport_building", "needs_assessment"]
            }
        },
        "customer_types": {
            "cautious": {
                "description": "Takes time to decide, needs lots of information",
                "preferences": ["rapport_building", "needs_assessment", "social_proof", "personalization"],
                "psychology_weights": {"interest": 0.3, "urgency": 0.2, "availability": 0.4, "trust": 0.25, "commitment": 0.2}
            },
            "price_shopper": {
                "description": "Very focused on getting the best deal",
                "preferences": ["value_proposition", "objection_handling", "incentive_offering"],
                "psychology_weights": {"interest": 0.45, "urgency": 0.35, "availability": 0.6, "trust": 0.35, "commitment": 0.35}
            },
            "ready_buyer": {
                "description": "Already knows what they want, ready to buy",
                "preferences": ["value_proposition", "appointment_booking", "incentive_offering"],
                "psychology_weights": {"interest": 0.6, "urgency": 0.5, "availability": 0.8, "trust": 0.45, "commitment": 0.5}
            },
            "research_buyer": {
                "description": "Wants to learn everything before deciding",
                "preferences": ["needs_assessment", "social_proof", "personalization", "follow_up"],
                "psychology_weights": {"interest": 0.3, "urgency": 0.2, "availability": 0.4, "trust": 0.25, "commitment": 0.2}
            },
            "impulse_buyer": {
                "description": "Makes quick decisions, easy to convince",
                "preferences": ["urgency_creation", "incentive_offering", "appointment_booking"],
                "psychology_weights": {"interest": 0.45, "urgency": 0.35, "availability": 0.6, "trust": 0.35, "commitment": 0.35}
            },
            "skeptical": {
                "description": "Hard to convince, needs lots of proof",
                "preferences": ["rapport_building", "objection_handling", "social_proof", "personalization"],
                "psychology_weights": {"interest": 0.6, "urgency": 0.5, "availability": 0.8, "trust": 0.45, "commitment": 0.5}
            }
        },
        "conversation_stages": {
            "early": {
                "description": "Initial contact and rapport building",
                "preferred_components": ["rapport_building", "needs_assessment", "personalization"]
            },
            "middle": {
                "description": "Value presentation and needs discovery",
                "preferred_components": ["needs_assessment", "value_proposition", "social_proof"]
            },
            "late": {
                "description": "Objection handling and closing preparation",
                "preferred_components": ["objection_handling", "urgency_creation", "incentive_offering"]
            },
            "closing": {
                "description": "Final push for appointment booking",
                "preferred_components": ["appointment_booking", "urgency_creation", "incentive_offering"]
            }
        }
    }
    
    with open("appointment_prompt_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created appointment_prompt_config.json")

def main():
    """Main training function."""
    print("🚀 Starting Appointment Booking Meta-Prompting RL Training")
    print("=" * 60)
    
    # Create configuration file
    create_appointment_prompt_config()
    
    # Configuration
    n_envs = int(os.environ.get("N_ENVS", 8))
    total_steps = int(os.environ.get("TOTAL_STEPS", 300_000))
    eval_freq = int(os.environ.get("EVAL_FREQ", 15000))
    
    print(f"📊 Training Configuration:")
    print(f"  - Parallel environments: {n_envs}")
    print(f"  - Total training steps: {total_steps:,}")
    print(f"  - Evaluation frequency: {eval_freq:,}")
    print()
    
    # Create vectorized environment
    env = make_vec_env(
        lambda: AppointmentPromptEnv("appointment_prompt_config.json"),
        n_envs=n_envs
    )
    
    # Create evaluation environment
    eval_env = AppointmentPromptEnv("appointment_prompt_config.json")
    
    # Initialize PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
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
        best_model_save_path="./best_models/",
        log_path="./eval_logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    print("🎯 Starting training...")
    print("   (The agent is learning optimal appointment booking prompt strategies)")
    print()
    
    # Train the model
    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        progress_bar=False
    )
    
    # Save the final model
    model.save("ppo_appointment_prompts")
    print("✅ Training complete!")
    print("📁 Model saved as: ppo_appointment_prompts.zip")
    print("📁 Best model saved in: ./best_models/")
    print("📊 Evaluation logs saved in: ./eval_logs/")
    print("📈 Tensorboard logs saved in: ./tensorboard_logs/")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_model(model, eval_env, num_episodes=5)

def test_model(model, env, num_episodes=5):
    """Test the trained model on a few episodes."""
    print(f"\n🔍 Testing model on {num_episodes} episodes...")
    
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        env.render()
        
        while step < 8:  # Max 8 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if done or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1} - Total Reward: {total_reward:.3f}")
        if 'prompt_effectiveness' in info:
            print(f"Prompt Effectiveness: {info['prompt_effectiveness']:.3f}")
        print(f"Components Used: {info.get('components_used', 0)}")
        print(f"Customer Type: {info.get('customer_type', 'Unknown')}")
        print(f"Conversation Stage: {info.get('conversation_stage', 'Unknown')}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\n📊 Average Reward: {avg_reward:.3f}")
    print(f"📊 Reward Std: {np.std(total_rewards):.3f}")

if __name__ == "__main__":
    main()

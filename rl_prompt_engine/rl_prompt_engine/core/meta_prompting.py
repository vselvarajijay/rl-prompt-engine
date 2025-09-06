#!/usr/bin/env python3
"""
Meta-Prompting Module

Core module for RL-powered meta-prompting system that generates
prompt templates for appointment booking.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from .appointment_prompt_env import AppointmentPromptEnv
from .appointment_prompt_generator import AppointmentPromptGenerator, AppointmentPromptDatabase
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

class MetaPromptingSystem:
    """Main class for the meta-prompting system."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the meta-prompting system.
        
        Args:
            model_path: Path to trained RL model (optional)
        """
        self.env = AppointmentPromptEnv()
        self.generator = AppointmentPromptGenerator()
        self.database = AppointmentPromptDatabase()
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained RL model."""
        try:
            self.model = PPO.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def train(self, total_timesteps: int = 10000, save_path: str = "ppo_meta_prompting"):
        """
        Train the RL model for meta-prompting.
        
        Args:
            total_timesteps: Number of training timesteps
            save_path: Path to save the trained model
        """
        print("🚀 Training RL model for meta-prompting...")
        
        vec_env = DummyVecEnv([lambda: self.env])
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
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        # Save the model
        model.save(save_path)
        self.model = model
        print(f"✅ Training completed. Model saved as {save_path}")
    
    def generate_strategy(self, customer_type: int, conversation_stage: int, 
                         urgency_level: int, customer_psychology: Optional[np.ndarray] = None) -> Dict:
        """
        Generate prompt strategy using RL agent.
        
        Args:
            customer_type: 0-5 (cautious, price_shopper, ready_buyer, etc.)
            conversation_stage: 0-3 (early, middle, late, closing)
            urgency_level: 0-2 (low, medium, high)
            customer_psychology: Customer state vector (optional)
            
        Returns:
            Dictionary with strategy information
        """
        if not self.model:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Create environment and set context
        obs, _ = self.env.reset()
        obs["customer_type"][0] = customer_type
        obs["conversation_stage"][0] = conversation_stage
        obs["urgency_level"][0] = urgency_level
        
        if customer_psychology is not None:
            obs["customer_psychology"] = customer_psychology
        
        # Generate strategy using RL agent
        selected_components = []
        step = 0
        max_steps = 6
        
        while step < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
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
        component_names = list(self.generator.config["prompt_components"].keys())
        selected_component_names = [component_names[i] for i in selected_components]
        
        # Calculate effectiveness
        template = self.generator.generate_prompt_template(
            customer_type=customer_type,
            conversation_stage=conversation_stage,
            urgency_level=urgency_level,
            selected_components=selected_components,
            customer_psychology=obs["customer_psychology"]
        )
        
        return {
            "selected_components": selected_component_names,
            "effectiveness_score": template.effectiveness_score,
            "customer_type": customer_type,
            "conversation_stage": conversation_stage,
            "urgency_level": urgency_level,
            "customer_psychology": obs["customer_psychology"].tolist()
        }
    
    def generate_prompt_template(self, customer_type: int, conversation_stage: int, 
                                urgency_level: int, customer_context: Optional[Dict] = None) -> str:
        """
        Generate a complete prompt template with parameters.
        
        Args:
            customer_type: 0-5 (cautious, price_shopper, ready_buyer, etc.)
            conversation_stage: 0-3 (early, middle, late, closing)
            urgency_level: 0-2 (low, medium, high)
            customer_context: Additional customer context (optional)
            
        Returns:
            Complete prompt template string
        """
        # Generate strategy
        strategy = self.generate_strategy(customer_type, conversation_stage, urgency_level)
        
        # Customer type names
        customer_types = ["cautious", "price_shopper", "ready_buyer", "research_buyer", "impulse_buyer", "skeptical"]
        conversation_stages = ["early", "middle", "late", "closing"]
        urgency_levels = ["low", "medium", "high"]
        
        customer_type_name = customer_types[customer_type]
        stage_name = conversation_stages[conversation_stage]
        urgency_name = urgency_levels[urgency_level]
        
        # Determine tone and approach
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
        if urgency_level == 2:  # high
            tone += " with a sense of urgency"
            time_reference = "today or tomorrow"
        elif urgency_level == 1:  # medium
            time_reference = "this week"
        else:  # low
            time_reference = "when convenient"
        
        # Build template based on strategy components
        template_parts = []
        
        if "rapport_building" in strategy["selected_components"]:
            template_parts.append("Hi {first_name}! I hope you're having a great day.")
        
        if "needs_assessment" in strategy["selected_components"]:
            template_parts.append("I'd love to understand what you're looking for in your next vehicle - what's most important to you when choosing a car?")
        
        if "value_proposition" in strategy["selected_components"]:
            template_parts.append("The {car_model} offers excellent value with {key_benefit}.")
        
        if "urgency_creation" in strategy["selected_components"]:
            template_parts.append("We have limited availability on the {car_model} this week.")
        
        if "incentive_offering" in strategy["selected_components"]:
            template_parts.append("I can get you a special financing rate of {finance_rate}% and {incentive}.")
        
        if "social_proof" in strategy["selected_components"]:
            template_parts.append("Many customers like you have been very satisfied with their {car_model} purchase.")
        
        if "objection_handling" in strategy["selected_components"]:
            template_parts.append("I understand your concerns about {concern} with the {car_model}.")
        
        if "appointment_booking" in strategy["selected_components"]:
            template_parts.append("Would you be available for a {appointment_duration} appointment {time_reference}?")
        
        if "personalization" in strategy["selected_components"]:
            template_parts.append("I'd love to help you find the perfect vehicle that meets your needs and budget.")
        
        # Ensure we have a complete message structure
        if not template_parts:
            template_parts.append("Hi {first_name}! I hope you're having a great day.")
            template_parts.append("I'd love to help you find the perfect vehicle that meets your needs and budget.")
            template_parts.append("Would you be available for a {appointment_duration} appointment {time_reference}?")
        
        # Create the full prompt template
        template = f"""You are an AI sales representative for {customer_type_name} customers. Generate an appointment booking message with the following specifications:

CUSTOMER PROFILE:
- Customer Type: {customer_type_name}
- Conversation Stage: {stage_name}
- Urgency Level: {urgency_name}
- Description: Customer needs {customer_type_name} approach

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
    
    def generate_templates_for_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Generate prompt templates for multiple scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            List of generated templates
        """
        templates = []
        
        for scenario in scenarios:
            template = self.generate_prompt_template(
                customer_type=scenario.get("customer_type", 0),
                conversation_stage=scenario.get("conversation_stage", 0),
                urgency_level=scenario.get("urgency_level", 0),
                customer_context=scenario.get("customer_context")
            )
            
            templates.append({
                "scenario": scenario,
                "template": template
            })
        
        return templates
    
    def save_templates(self, templates: List[Dict], filename: str = "prompt_templates.json"):
        """Save templates to file."""
        with open(filename, "w") as f:
            json.dump(templates, f, indent=2)
        print(f"✅ Saved {len(templates)} templates to {filename}")
    
    def load_templates(self, filename: str = "prompt_templates.json") -> List[Dict]:
        """Load templates from file."""
        with open(filename, "r") as f:
            return json.load(f)

# Convenience functions
def create_system(model_path: Optional[str] = None) -> MetaPromptingSystem:
    """Create a new MetaPromptingSystem instance."""
    return MetaPromptingSystem(model_path)

def train_system(total_timesteps: int = 10000, save_path: str = "ppo_meta_prompting") -> MetaPromptingSystem:
    """Train a new system and return it."""
    system = MetaPromptingSystem()
    system.train(total_timesteps, save_path)
    return system

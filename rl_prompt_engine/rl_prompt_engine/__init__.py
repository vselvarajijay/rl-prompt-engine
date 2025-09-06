"""
RL Prompt Engine

A reinforcement learning-powered system for generating prompt templates
for appointment booking conversations.

Core Components:
- MetaPromptingSystem: Main system class
- AppointmentPromptEnv: RL environment
- AppointmentPromptGenerator: Template generator
- AppointmentPromptDatabase: Template storage

Usage:
    from rl_prompt_engine import MetaPromptingSystem
    
    # Create system
    system = MetaPromptingSystem()
    
    # Train model
    system.train(total_timesteps=10000)
    
    # Generate template
    template = system.generate_prompt_template(
        customer_type=0,  # cautious
        conversation_stage=0,  # early
        urgency_level=0  # low
    )
"""

from .core.meta_prompting import MetaPromptingSystem, create_system, train_system
from .core.appointment_prompt_env import AppointmentPromptEnv
from .core.appointment_prompt_generator import AppointmentPromptGenerator, AppointmentPromptDatabase

__version__ = "0.1.0"
__author__ = "Vijay Selvaraj"

__all__ = [
    "MetaPromptingSystem",
    "create_system", 
    "train_system",
    "AppointmentPromptEnv",
    "AppointmentPromptGenerator",
    "AppointmentPromptDatabase"
]

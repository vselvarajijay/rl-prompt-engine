"""
RL Prompt Engine

A reinforcement learning-powered system for generating prompt templates
for any use case. Fully configurable and generic.

Core Components:
- PromptSystem: Main system class
- PromptEnv: RL environment
- PromptGenerator: Template generator
- PromptTemplate: Template representation

Usage:
    from rl_prompt_engine import PromptSystem
    
    # Create system
    system = PromptSystem(config_file="configs/generic_config.json")
    
    # Train model
    system.train(total_timesteps=10000)
    
    # Generate template
    template = system.generate_template(
        context_type=0,  # new_customer
        conversation_stage=0,  # opening
        urgency_level=0  # low
    )
"""

from .core.prompt_engine import PromptEngine
from .core.prompt_env import PromptEnv

__version__ = "2.0.0"
__author__ = "Vijay Selvaraj"

__all__ = [
    "PromptEngine",
    "PromptEnv"
]

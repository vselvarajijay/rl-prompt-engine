#!/usr/bin/env python3
"""
RL Prompt Engine

A clean, simple RL system for learning prompt construction strategies using PPO.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import json
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from .logging_config import setup_logging
from .prompt_env import PromptEnv
from .template_loader import TemplateLoader

class PromptEngine:
    """Main RL Prompt Engine class."""
    
    def __init__(self, config_file: str = "configs/generic_config.json"):
        """Initialize the prompt engine."""
        # Setup logging
        self.loggers = setup_logging()
        self.training_logger = self.loggers['training']
        
        # Initialize environment
        self.env = PromptEnv(config_file)
        self.config_file = config_file
        self.config = self.env.config  # Store config for access
        self.model = None
        
        # Initialize template loader
        self.template_loader = TemplateLoader()
    
    def train(self, 
              total_timesteps: int = 10000, 
              save_path: str = "models/prompt_engine_model",
              learning_rate: float = 0.0003,
              n_steps: int = 2048,
              batch_size: int = 512,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              ent_coef: float = 0.1,
              vf_coef: float = 0.5,
              max_grad_norm: float = 0.5,
              verbose: int = 1) -> PPO:
        """Train the PPO model."""
        self.training_logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: PromptEnv(self.config_file)])
        
        # Initialize PPO model with MlpPolicy for flattened observations
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=save_path,
            log_path=f"{save_path}_logs",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.training_logger.info("Starting training...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.training_logger.info(f"Saving model to {save_path}")
        self.model.save(save_path)
        
        self.training_logger.info("Training completed successfully")
        print(f"✅ Training completed. Model saved as {save_path}")
        
        return self.model
    
    def load_model(self, model_path: str) -> PPO:
        """Load a trained PPO model."""
        try:
            # Check if it's a directory, if so, look for the zip file
            if os.path.isdir(model_path):
                zip_path = f"{model_path}.zip"
                if os.path.exists(zip_path):
                    model_path = zip_path
                else:
                    raise ValueError(f"Model directory found but no zip file at {zip_path}")
            elif not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            self.model = PPO.load(model_path)
            self.training_logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            self.training_logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_strategy(self, 
                         context_type: int, 
                         conversation_stage: int, 
                         urgency_level: int) -> List[str]:
        """Generate a prompt strategy using the trained model."""
        if not self.model:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Reset environment with specific context
        obs, _ = self.env.reset(options={
            "context_type": context_type,
            "conversation_stage": conversation_stage,
            "urgency_level": urgency_level
        })
        
        # Generate strategy using PPO agent
        selected_components = []
        step = 0
        max_steps = 10
        
        while step < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            # Handle both scalar and array actions
            try:
                if hasattr(action, '__len__') and len(action) > 0:
                    action = int(action[0])
                else:
                    action = int(action)
            except (TypeError, IndexError):
                # Fallback for edge cases
                action = int(action)
            
            if action == len(self.env.prompt_components):  # Finish action
                break
            
            if action not in selected_components:
                selected_components.append(action)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1
            
            if terminated or truncated:
                break
        
        # Return component names
        return [self.env.prompt_components[i] for i in selected_components]
    
    def generate_template(self, 
                         context_type: int, 
                         conversation_stage: int, 
                         urgency_level: int,
                         custom_variables: Optional[Dict[str, str]] = None) -> str:
        """Generate a prompt template."""
        # Generate strategy
        strategy = self.generate_strategy(context_type, conversation_stage, urgency_level)
        
        # Get context information
        context_type_name = self.env.context_types[context_type]
        stage_name = self.env.conversation_stages[conversation_stage]
        urgency_name = self.env.urgency_levels[urgency_level]
        
        context_config = self.env.config["context_types"][context_type_name]
        stage_config = self.env.config["conversation_stages"][stage_name]
        urgency_config = self.env.config["urgency_levels"][urgency_name]
        
        # Require custom variables to be provided
        if not custom_variables:
            raise ValueError("custom_variables must be provided. No default values allowed.")
        
        # Generate template parts
        template_parts = []
        for component_name in strategy:
            if component_name in self.env.config["prompt_components"]:
                component_config = self.env.config["prompt_components"][component_name]
                template_part = component_config["template"]
                
                # Fill in variables
                for key, value in custom_variables.items():
                    template_part = template_part.replace(f"{{{key}}}", value)
                
                template_parts.append(template_part)
        
        # Combine all parts
        full_template = "\n\n".join(template_parts)
        
        # Create meta-prompt using template
        template_variables = {
            'context_type_name': context_type_name,
            'stage_name': stage_name,
            'urgency_name': urgency_name,
            'context_description': context_config['description'],
            'context_tone': context_config['tone'],
            'context_approach': context_config['approach'],
            'urgency_time_reference': urgency_config['time_reference'],
            'full_template': full_template
        }
        
        try:
            meta_prompt = self.template_loader.render_template('meta_prompt_template', template_variables)
        except FileNotFoundError:
            # Fail gracefully if template not found
            self.training_logger.error("Meta prompt template not found. Create rl_prompt_engine/templates/meta_prompt_template.md")
            raise FileNotFoundError(
                "Meta prompt template not found. Please create rl_prompt_engine/templates/meta_prompt_template.md"
            )
        
        return meta_prompt
    
    def evaluate_strategy(self, 
                         context_type: int, 
                         conversation_stage: int, 
                         urgency_level: int,
                         strategy: List[str]) -> Dict[str, float]:
        """Evaluate a prompt strategy."""
        # Reset environment
        obs, _ = self.env.reset(options={
            "context_type": context_type,
            "conversation_stage": conversation_stage,
            "urgency_level": urgency_level
        })
        
        # Convert strategy to component indices
        component_indices = []
        for component_name in strategy:
            if component_name in self.env.prompt_components:
                component_indices.append(self.env.prompt_components.index(component_name))
        
        # Calculate rewards for each component
        total_reward = 0.0
        for component_idx in component_indices:
            reward = self.env._calculate_component_reward(component_idx)
            total_reward += reward
        
        # Calculate final reward
        if component_indices:
            self.env.selected_components = component_indices
            final_reward = self.env._calculate_final_reward()
        else:
            final_reward = 0.0
        
        return {
            "total_reward": total_reward,
            "final_reward": final_reward,
            "component_count": len(strategy),
            "effectiveness": final_reward
        }
    
    def get_available_contexts(self) -> Dict[str, List[str]]:
        """Get available context information."""
        return {
            "context_types": self.env.context_types,
            "conversation_stages": self.env.conversation_stages,
            "urgency_levels": self.env.urgency_levels,
            "prompt_components": self.env.prompt_components
        }
    
    def get_available_templates(self) -> list:
        """Get list of available template files."""
        return self.template_loader.get_available_templates()
    
    def validate_template(self, template_name: str) -> Dict[str, list]:
        """Validate a template and return variable information."""
        return self.template_loader.validate_template(template_name)
    
    def load_template(self, template_name: str) -> str:
        """Load a template file directly."""
        return self.template_loader.load_template(template_name)

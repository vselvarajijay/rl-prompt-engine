#!/usr/bin/env python3
"""
Generic RL Prompt System

Main system class for RL-powered prompt generation.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .prompt_env import PromptEnv
from .prompt_generator import PromptGenerator, PromptTemplate
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from .logging_config import setup_logging, get_logger
from .training_callback import DetailedTrainingCallback

class PromptSystem:
    """Main class for the RL prompt generation system."""
    
    def __init__(self, model_path: Optional[str] = None, config_file: str = "configs/default_config.json"):
        """
        Initialize the prompt system.
        
        Args:
            model_path: Path to trained RL model (optional)
            config_file: Path to configuration file
        """
        # Setup logging
        self.loggers = setup_logging()
        self.training_logger = self.loggers['training']
        self.env_logger = self.loggers['environment']
        self.model_logger = self.loggers['model']
        
        # Initialize components
        self.env = PromptEnv(config_file)
        self.generator = PromptGenerator(config_file)
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained RL model."""
        try:
            # Check if it's a directory with best_model.zip
            from pathlib import Path
            model_path_obj = Path(model_path)
            if model_path_obj.is_dir() and (model_path_obj / "best_model.zip").exists():
                model_path = str(model_path_obj / "best_model.zip")
            
            self.model = PPO.load(model_path)
            self.model_logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.model_logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self, total_timesteps: int = 10000, save_path: str = "models/ppo_prompt_system"):
        """Train the RL model."""
        self.training_logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=512,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path=save_path,
            log_path=f"{save_path}_logs",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        detailed_callback = DetailedTrainingCallback(log_freq=1000)
        
        callback = CallbackList([eval_callback, detailed_callback])
        
        # Train the model
        self.training_logger.info("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        # Save the model
        self.training_logger.info(f"Saving model to {save_path}")
        model.save(save_path)
        self.model = model
        self.training_logger.info("Training completed successfully")
        print(f"✅ Training completed. Model saved as {save_path}")
    
    def generate_template(self, 
                         context_type: int, 
                         conversation_stage: int, 
                         urgency_level: int,
                         custom_variables: Optional[Dict[str, str]] = None) -> PromptTemplate:
        """
        Generate a prompt template using the trained model.
        
        Args:
            context_type: Index of context type
            conversation_stage: Index of conversation stage
            urgency_level: Index of urgency level
            custom_variables: Optional custom variables for template filling
            
        Returns:
            Generated PromptTemplate object
        """
        if not self.model:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        # Reset environment with specific context
        obs, _ = self.env.reset(options={
            "context_type": context_type,
            "conversation_stage": conversation_stage,
            "urgency_level": urgency_level
        })
        
        # Generate strategy using RL agent
        selected_components = []
        step = 0
        max_steps = 10
        
        while step < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            
            if action == len(self.env.prompt_components):  # Finish action
                break
            
            if action not in selected_components:
                selected_components.append(action)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1
            
            if terminated or truncated:
                break
        
        # Generate template
        template = self.generator.generate_template(
            context_type=context_type,
            conversation_stage=conversation_stage,
            urgency_level=urgency_level,
            selected_components=selected_components,
            custom_variables=custom_variables
        )
        
        return template
    
    def generate_meta_prompt(self, 
                           context_type: int, 
                           conversation_stage: int, 
                           urgency_level: int,
                           custom_variables: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a meta-prompt template for use with LLMs.
        
        Args:
            context_type: Index of context type
            conversation_stage: Index of conversation stage
            urgency_level: Index of urgency level
            custom_variables: Optional custom variables for template filling
            
        Returns:
            Meta-prompt string ready for use with LLMs
        """
        template = self.generate_template(
            context_type=context_type,
            conversation_stage=conversation_stage,
            urgency_level=urgency_level,
            custom_variables=custom_variables
        )
        
        return self.generator.generate_meta_prompt(template)
    
    def generate_templates_for_scenarios(self, scenarios: List[Dict]) -> List[PromptTemplate]:
        """
        Generate templates for multiple scenarios.
        
        Args:
            scenarios: List of scenario dictionaries with context_type, conversation_stage, urgency_level
            
        Returns:
            List of generated PromptTemplate objects
        """
        templates = []
        
        for scenario in scenarios:
            try:
                template = self.generate_template(
                    context_type=scenario["context_type"],
                    conversation_stage=scenario["conversation_stage"],
                    urgency_level=scenario["urgency_level"],
                    custom_variables=scenario.get("custom_variables")
                )
                templates.append(template)
            except Exception as e:
                self.env_logger.error(f"Failed to generate template for scenario {scenario}: {e}")
        
        return templates
    
    def get_context_info(self) -> Dict[str, List[str]]:
        """Get available context information."""
        return {
            "context_types": self.env.context_types,
            "conversation_stages": self.env.conversation_stages,
            "urgency_levels": self.env.urgency_levels,
            "prompt_components": self.env.prompt_components
        }
    
    def evaluate_template(self, template: PromptTemplate) -> Dict[str, float]:
        """Evaluate a generated template."""
        return {
            "effectiveness_score": template.effectiveness_score,
            "component_count": len(template.components),
            "context_match": 1.0 if template.context_type in self.env.context_types else 0.0,
            "stage_match": 1.0 if template.conversation_stage in self.env.conversation_stages else 0.0
        }

# Convenience functions
def create_system(model_path: Optional[str] = None, config_file: str = "configs/default_config.json") -> PromptSystem:
    """Create a new PromptSystem instance."""
    return PromptSystem(model_path=model_path, config_file=config_file)

def train_system(config_file: str = "configs/default_config.json", 
                total_timesteps: int = 10000, 
                save_path: str = "models/ppo_prompt_system") -> PromptSystem:
    """Train a new PromptSystem."""
    system = PromptSystem(config_file=config_file)
    system.train(total_timesteps=total_timesteps, save_path=save_path)
    return system

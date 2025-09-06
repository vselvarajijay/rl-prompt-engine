#!/usr/bin/env python3
"""
Generic RL Prompt Environment

A configurable RL environment for learning optimal prompt construction strategies
for any use case. The environment is driven entirely by configuration files.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import json
import random
from pathlib import Path

class PromptEnv(gym.Env):
    """
    Generic RL Environment for Learning Prompt Construction Strategies
    
    The agent learns to construct optimal prompts by:
    1. Observing context (user type, conversation stage, urgency, etc.)
    2. Selecting prompt components to include
    3. Receiving feedback on prompt effectiveness
    
    Everything is configurable via JSON config files.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config_file: str = "configs/default_config.json"):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Extract configuration parameters
        self.prompt_components = list(self.config["prompt_components"].keys())
        self.context_types = list(self.config["context_types"].keys())
        self.conversation_stages = list(self.config["conversation_stages"].keys())
        self.urgency_levels = list(self.config["urgency_levels"].keys())
        
        # Environment parameters
        self.max_prompt_length = self.config.get("max_prompt_length", 6)
        self.max_turns = self.config.get("max_turns", 10)
        
        # State variables
        self.selected_components = []
        self.turn = 0
        self.current_context_type = 0
        self.current_conversation_stage = 0
        self.current_urgency_level = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.prompt_components) + 1)  # +1 for finish action
        self.observation_space = spaces.Dict({
            "context_type": spaces.Discrete(len(self.context_types)),
            "conversation_stage": spaces.Discrete(len(self.conversation_stages)),
            "urgency_level": spaces.Discrete(len(self.urgency_levels)),
            "selected_components": spaces.MultiBinary(len(self.prompt_components)),
            "turn": spaces.Box(0, self.max_turns, shape=(1,), dtype=np.float32)
        })
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.selected_components = []
        self.turn = 0
        
        # Randomly select context (can be overridden in options)
        if options and "context_type" in options:
            self.current_context_type = options["context_type"]
        else:
            self.current_context_type = self.np_random.integers(0, len(self.context_types))
        
        if options and "conversation_stage" in options:
            self.current_conversation_stage = options["conversation_stage"]
        else:
            self.current_conversation_stage = self.np_random.integers(0, len(self.conversation_stages))
        
        if options and "urgency_level" in options:
            self.current_urgency_level = options["urgency_level"]
        else:
            self.current_urgency_level = self.np_random.integers(0, len(self.urgency_levels))
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.turn += 1
        
        # Check if action is finish (last action)
        if action == len(self.prompt_components):
            # Finish prompt construction
            terminated = True
            reward = self._calculate_final_reward()
            info = {
                "prompt_effectiveness": reward,
                "components_used": len(self.selected_components),
                "turns_taken": self.turn
            }
        else:
            # Select a component
            component_idx = int(action)
            component_name = self.prompt_components[component_idx]
            
            if component_idx in self.selected_components:
                # Invalid action - already selected
                reward = -0.1
                terminated = False
                info = {"error": "Component already selected"}
            else:
                # Valid action - add component
                self.selected_components.append(component_idx)
                reward = self._calculate_component_reward(component_idx)
                terminated = False
                info = {"component_added": component_name}
        
        # Check for truncation
        if self.turn >= self.max_turns:
            terminated = True
            if not self.selected_components:  # No components selected
                reward = 0.0
            else:
                reward = self._calculate_final_reward()
        
        return self._get_observation(), reward, terminated, False, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        selected_binary = np.zeros(len(self.prompt_components), dtype=np.float32)
        for idx in self.selected_components:
            selected_binary[idx] = 1.0
        
        return {
            "context_type": np.array([self.current_context_type], dtype=np.int32),
            "conversation_stage": np.array([self.current_conversation_stage], dtype=np.int32),
            "urgency_level": np.array([self.current_urgency_level], dtype=np.int32),
            "selected_components": selected_binary,
            "turn": np.array([self.turn], dtype=np.float32)
        }
    
    def _calculate_component_reward(self, component_idx: int) -> float:
        """Calculate reward for selecting a specific component."""
        component_name = self.prompt_components[component_idx]
        component_config = self.config["prompt_components"][component_name]
        
        # Base effectiveness
        effectiveness = component_config.get("effectiveness", 0.5)
        
        # Context type bonus
        context_type_name = self.context_types[self.current_context_type]
        context_config = self.config["context_types"][context_type_name]
        if component_name in context_config.get("preferred_components", []):
            effectiveness *= 1.2  # 20% bonus
        
        # Stage bonus
        stage_name = self.conversation_stages[self.current_conversation_stage]
        stage_config = self.config["conversation_stages"][stage_name]
        if component_name in stage_config.get("preferred_components", []):
            effectiveness *= 1.1  # 10% bonus
        
        # Urgency bonus
        urgency_name = self.urgency_levels[self.current_urgency_level]
        urgency_config = self.config["urgency_levels"][urgency_name]
        if urgency_name == "high" and component_name in urgency_config.get("preferred_components", []):
            effectiveness *= 1.15  # 15% bonus
        
        return effectiveness * 0.1  # Scale down for step rewards
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward for the constructed prompt."""
        if not self.selected_components:
            return 0.0
        
        # Base effectiveness from components
        total_effectiveness = 0.0
        for component_idx in self.selected_components:
            component_name = self.prompt_components[component_idx]
            component_config = self.config["prompt_components"][component_name]
            total_effectiveness += component_config.get("effectiveness", 0.5)
        
        avg_effectiveness = total_effectiveness / len(self.selected_components)
        
        # Efficiency bonus (fewer components = more efficient)
        efficiency_bonus = max(0, (self.max_prompt_length - len(self.selected_components)) * 0.05)
        
        # Context matching bonus
        context_matching_bonus = self._calculate_context_matching_bonus()
        
        # Combine factors
        final_score = avg_effectiveness + efficiency_bonus + context_matching_bonus
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _calculate_context_matching_bonus(self) -> float:
        """Calculate bonus for how well components match the context."""
        if not self.selected_components:
            return 0.0
        
        context_type_name = self.context_types[self.current_context_type]
        stage_name = self.conversation_stages[self.current_conversation_stage]
        
        context_config = self.config["context_types"][context_type_name]
        stage_config = self.config["conversation_stages"][stage_name]
        
        preferred_components = set(context_config.get("preferred_components", [])) | \
                             set(stage_config.get("preferred_components", []))
        
        selected_component_names = [self.prompt_components[i] for i in self.selected_components]
        matching_components = set(selected_component_names) & preferred_components
        
        return len(matching_components) * 0.1  # 0.1 bonus per matching component
    
    def render(self, mode: str = "human") -> str:
        """Render the current state."""
        component_names = [self.prompt_components[i] for i in self.selected_components]
        context_type = self.context_types[self.current_context_type]
        stage = self.conversation_stages[self.current_conversation_stage]
        urgency = self.urgency_levels[self.current_urgency_level]
        
        return f"Prompt: {' + '.join(component_names)} | Context: {context_type} | Stage: {stage} | Urgency: {urgency}"
    
    def get_selected_components(self) -> List[str]:
        """Get names of selected components."""
        return [self.prompt_components[i] for i in self.selected_components]
    
    def get_context_info(self) -> Dict[str, str]:
        """Get current context information."""
        return {
            "context_type": self.context_types[self.current_context_type],
            "conversation_stage": self.conversation_stages[self.current_conversation_stage],
            "urgency_level": self.urgency_levels[self.current_urgency_level]
        }

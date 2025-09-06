#!/usr/bin/env python3
"""
Appointment Booking Meta-Prompting RL Environment

This environment trains an RL agent to learn optimal prompt construction strategies
for generating appointment booking prompts that work with different customer types
and conversation contexts.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import json
import random
from pathlib import Path

# Constants
N_PROMPT_COMPONENTS = 10  # Number of different prompt components for appointment booking
MAX_PROMPT_LENGTH = 6     # Maximum number of components in a single prompt
N_CUSTOMER_TYPES = 6      # Different customer personas (from original system)
N_CONVERSATION_STAGES = 4 # Early, Middle, Late, Closing
N_URGENCY_LEVELS = 3      # Low, Medium, High

class AppointmentPromptEnv(gym.Env):
    """
    RL Environment for Learning Appointment Booking Prompt Strategies
    
    The agent learns to construct optimal prompts by:
    1. Observing customer context (type, conversation stage, urgency, psychology)
    2. Selecting prompt components to include in the booking prompt
    3. Arranging them in an effective order
    4. Receiving feedback on prompt effectiveness
    
    Observation Space:
    - customer_type: Discrete(6) - Type of customer (cautious, price shopper, etc.)
    - conversation_stage: Discrete(4) - Early, Middle, Late, Closing
    - urgency_level: Discrete(3) - Low, Medium, High urgency
    - customer_psychology: Box(5,) - [interest, urgency, availability, trust, commitment]
    - prompt_so_far: Box(MAX_PROMPT_LENGTH,) - Current prompt construction state
    - turn: Box(1,) - Current step in prompt construction
    
    Action Space:
    - component_selection: Discrete(N_PROMPT_COMPONENTS) - Which component to add
    - position_selection: Discrete(MAX_PROMPT_LENGTH) - Where to place it
    - finish_prompt: Discrete(2) - Whether to finish construction
    
    Reward:
    - Based on prompt effectiveness score (0-1)
    - Bonus for efficient construction (fewer steps)
    - Penalty for poor component combinations
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, config_file: str = "appointment_prompt_config.json"):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "customer_type": spaces.Discrete(N_CUSTOMER_TYPES),
            "conversation_stage": spaces.Discrete(N_CONVERSATION_STAGES),
            "urgency_level": spaces.Discrete(N_URGENCY_LEVELS),
            "customer_psychology": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            "prompt_so_far": spaces.Box(low=0.0, high=1.0, shape=(MAX_PROMPT_LENGTH,), dtype=np.float32),
            "turn": spaces.Box(low=0, high=MAX_PROMPT_LENGTH, shape=(1,), dtype=np.float32),
        })
        
        # Define action space - flattened for PPO compatibility
        # Action is a single integer: component * 100 + position * 10 + finish
        self.action_space = spaces.Discrete(N_PROMPT_COMPONENTS * MAX_PROMPT_LENGTH * 2)
        
        # Initialize prompt components
        self.prompt_components = self._initialize_prompt_components()
        
        # Initialize customer types (from original system)
        self.customer_types = self._initialize_customer_types()
        
        # Initialize conversation stages
        self.conversation_stages = self._initialize_conversation_stages()
        
        self.reset()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        config_path = Path(config_file)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = self._get_default_config()
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for appointment booking prompts."""
        return {
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
    
    def _initialize_prompt_components(self) -> List[Dict[str, Any]]:
        """Initialize prompt components from config."""
        components = []
        for name, config in self.config["prompt_components"].items():
            components.append({
                "name": name,
                "description": config["description"],
                "effectiveness": config["effectiveness"],
                "stage_effectiveness": config["stage_effectiveness"],
                "compatibility": config["compatibility"]
            })
        return components
    
    def _initialize_customer_types(self) -> List[Dict[str, Any]]:
        """Initialize customer types from config."""
        types = []
        for name, config in self.config["customer_types"].items():
            types.append({
                "name": name,
                "description": config["description"],
                "preferences": config["preferences"],
                "psychology_weights": config["psychology_weights"]
            })
        return types
    
    def _initialize_conversation_stages(self) -> List[Dict[str, Any]]:
        """Initialize conversation stages from config."""
        stages = []
        for name, config in self.config["conversation_stages"].items():
            stages.append({
                "name": name,
                "description": config["description"],
                "preferred_components": config["preferred_components"]
            })
        return stages
    
    def reset(self, seed: int = None, options: Dict = None):
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        # Randomly select customer context
        self.customer_type = np.random.randint(0, N_CUSTOMER_TYPES)
        self.conversation_stage = np.random.randint(0, N_CONVERSATION_STAGES)
        self.urgency_level = np.random.randint(0, N_URGENCY_LEVELS)
        
        # Initialize customer psychology based on customer type
        customer_type_config = self.customer_types[self.customer_type]
        psychology_weights = customer_type_config["psychology_weights"]
        
        # Add some randomness to customer psychology
        self.customer_psychology = np.array([
            psychology_weights["interest"] + np.random.normal(0, 0.1),
            psychology_weights["urgency"] + np.random.normal(0, 0.1),
            psychology_weights["availability"] + np.random.normal(0, 0.1),
            psychology_weights["trust"] + np.random.normal(0, 0.1),
            psychology_weights["commitment"] + np.random.normal(0, 0.1)
        ], dtype=np.float32)
        self.customer_psychology = np.clip(self.customer_psychology, 0.0, 1.0)
        
        # Initialize prompt construction state
        self.prompt_so_far = np.zeros(MAX_PROMPT_LENGTH, dtype=np.float32)
        self.turn = 0
        self.selected_components = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step of prompt construction."""
        # Decode flattened action
        action = int(action[0]) if hasattr(action, '__len__') else int(action)
        finish = action % 2
        position = (action // 2) % MAX_PROMPT_LENGTH
        component_idx = (action // (2 * MAX_PROMPT_LENGTH)) % N_PROMPT_COMPONENTS
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if finish == 1 or self.turn >= MAX_PROMPT_LENGTH:
            # Finish prompt construction
            terminated = True
            reward = self._calculate_prompt_effectiveness()
            info = {
                "prompt_effectiveness": reward,
                "components_used": len(self.selected_components),
                "efficiency": len(self.selected_components) / max(self.turn, 1),
                "customer_type": self.customer_types[self.customer_type]["name"],
                "conversation_stage": self.conversation_stages[self.conversation_stage]["name"]
            }
        else:
            # Add component to prompt
            if component_idx not in self.selected_components and position < MAX_PROMPT_LENGTH:
                self.selected_components.append(component_idx)
                self.prompt_so_far[position] = 1.0
                
                # Calculate immediate reward for component selection
                reward = self._calculate_component_reward(component_idx)
                
                # Check for compatibility bonuses/penalties
                if len(self.selected_components) > 1:
                    reward += self._calculate_compatibility_reward()
            else:
                # Invalid action - penalty
                reward = -0.1
            
            self.turn += 1
            
            # Check for truncation
            if self.turn >= MAX_PROMPT_LENGTH:
                truncated = True
                reward = self._calculate_prompt_effectiveness()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        return {
            "customer_type": np.array([self.customer_type], dtype=np.int32),
            "conversation_stage": np.array([self.conversation_stage], dtype=np.int32),
            "urgency_level": np.array([self.urgency_level], dtype=np.int32),
            "customer_psychology": self.customer_psychology,
            "prompt_so_far": self.prompt_so_far,
            "turn": np.array([self.turn], dtype=np.float32)
        }
    
    def _calculate_component_reward(self, component_idx: int) -> float:
        """Calculate reward for selecting a specific component."""
        component = self.prompt_components[component_idx]
        customer_type_name = self.customer_types[self.customer_type]["name"]
        conversation_stage_name = self.conversation_stages[self.conversation_stage]["name"]
        
        # Base effectiveness based on customer type
        base_effectiveness = component["effectiveness"][customer_type_name]
        
        # Stage effectiveness bonus
        stage_effectiveness = component["stage_effectiveness"][conversation_stage_name]
        combined_effectiveness = (base_effectiveness + stage_effectiveness) / 2
        
        # Check if component matches customer preferences
        customer_preferences = self.customer_types[self.customer_type]["preferences"]
        if component["name"] in customer_preferences:
            combined_effectiveness *= 1.2  # 20% bonus for preferred components
        
        # Check if component matches conversation stage preferences
        stage_preferences = self.conversation_stages[self.conversation_stage]["preferred_components"]
        if component["name"] in stage_preferences:
            combined_effectiveness *= 1.1  # 10% bonus for stage-matching components
        
        # Urgency level bonus
        urgency_levels = ["low", "medium", "high"]
        if urgency_levels[self.urgency_level] == "high" and component["name"] in ["urgency_creation", "appointment_booking"]:
            combined_effectiveness *= 1.15  # 15% bonus for urgency components when urgency is high
        
        return combined_effectiveness * 0.1  # Scale down for step rewards
    
    def _calculate_compatibility_reward(self) -> float:
        """Calculate reward/penalty for component compatibility."""
        if len(self.selected_components) < 2:
            return 0.0
        
        compatibility_score = 0.0
        total_pairs = 0
        
        for i, comp1_idx in enumerate(self.selected_components):
            for j, comp2_idx in enumerate(self.selected_components):
                if i != j:
                    comp1 = self.prompt_components[comp1_idx]
                    comp2 = self.prompt_components[comp2_idx]
                    
                    # Check if components are compatible
                    if comp2["name"] in comp1["compatibility"]:
                        compatibility_score += 1.0
                    else:
                        compatibility_score -= 0.3  # Smaller penalty for incompatible components
                    
                    total_pairs += 1
        
        if total_pairs > 0:
            return (compatibility_score / total_pairs) * 0.05
        return 0.0
    
    def _calculate_prompt_effectiveness(self) -> float:
        """Calculate final effectiveness score for the constructed prompt."""
        if not self.selected_components:
            return 0.0
        
        # Base effectiveness from individual components
        total_effectiveness = 0.0
        customer_type_name = self.customer_types[self.customer_type]["name"]
        conversation_stage_name = self.conversation_stages[self.conversation_stage]["name"]
        
        for comp_idx in self.selected_components:
            component = self.prompt_components[comp_idx]
            base_effectiveness = component["effectiveness"][customer_type_name]
            stage_effectiveness = component["stage_effectiveness"][conversation_stage_name]
            total_effectiveness += (base_effectiveness + stage_effectiveness) / 2
        
        # Average effectiveness
        avg_effectiveness = total_effectiveness / len(self.selected_components)
        
        # Customer preference bonus
        customer_preferences = self.customer_types[self.customer_type]["preferences"]
        preference_bonus = 0.0
        for comp_idx in self.selected_components:
            component = self.prompt_components[comp_idx]
            if component["name"] in customer_preferences:
                preference_bonus += 0.1
        
        # Stage matching bonus
        stage_preferences = self.conversation_stages[self.conversation_stage]["preferred_components"]
        stage_bonus = 0.0
        for comp_idx in self.selected_components:
            component = self.prompt_components[comp_idx]
            if component["name"] in stage_preferences:
                stage_bonus += 0.1
        
        # Compatibility bonus
        compatibility_bonus = self._calculate_compatibility_reward() * 10
        
        # Efficiency bonus (fewer components = more efficient)
        efficiency_bonus = max(0, (MAX_PROMPT_LENGTH - len(self.selected_components)) * 0.05)
        
        # Psychology alignment bonus
        psychology_bonus = self._calculate_psychology_alignment_bonus()
        
        # Combine all factors
        final_score = avg_effectiveness + preference_bonus + stage_bonus + compatibility_bonus + efficiency_bonus + psychology_bonus
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _calculate_psychology_alignment_bonus(self) -> float:
        """Calculate bonus based on how well components align with customer psychology."""
        if not self.selected_components:
            return 0.0
        
        psychology = self.customer_psychology
        interest, urgency, availability, trust, commitment = psychology
        
        bonus = 0.0
        
        for comp_idx in self.selected_components:
            component = self.prompt_components[comp_idx]
            comp_name = component["name"]
            
            # High interest customers benefit from value proposition and social proof
            if interest > 0.6 and comp_name in ["value_proposition", "social_proof"]:
                bonus += 0.05
            
            # High urgency customers benefit from urgency creation and appointment booking
            if urgency > 0.6 and comp_name in ["urgency_creation", "appointment_booking"]:
                bonus += 0.05
            
            # High availability customers benefit from appointment booking
            if availability > 0.6 and comp_name == "appointment_booking":
                bonus += 0.05
            
            # High trust customers benefit from direct approaches
            if trust > 0.6 and comp_name in ["appointment_booking", "value_proposition"]:
                bonus += 0.05
            
            # High commitment customers benefit from appointment booking and follow-up
            if commitment > 0.6 and comp_name in ["appointment_booking", "follow_up"]:
                bonus += 0.05
        
        return min(bonus, 0.2)  # Cap the psychology bonus
    
    def render(self, mode: str = "human"):
        """Render the current state."""
        if mode == "human":
            print(f"\n=== Appointment Booking Prompt Construction ===")
            print(f"Customer Type: {self.customer_types[self.customer_type]['name']}")
            print(f"Conversation Stage: {self.conversation_stages[self.conversation_stage]['name']}")
            print(f"Urgency Level: {['Low', 'Medium', 'High'][self.urgency_level]}")
            print(f"Customer Psychology: Interest={self.customer_psychology[0]:.2f}, Trust={self.customer_psychology[3]:.2f}, Commitment={self.customer_psychology[4]:.2f}")
            print(f"Turn: {self.turn}")
            print(f"Selected Components: {[self.prompt_components[i]['name'] for i in self.selected_components]}")
            print(f"Prompt So Far: {self.prompt_so_far}")
    
    def get_prompt_template(self) -> str:
        """Generate the actual prompt template from selected components."""
        if not self.selected_components:
            return "No prompt constructed yet."
        
        # This would generate the actual prompt template
        # For now, return a simple representation
        component_names = [self.prompt_components[i]['name'] for i in self.selected_components]
        return f"Appointment Booking Prompt: {' + '.join(component_names)}"

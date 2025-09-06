#!/usr/bin/env python3
"""
Easy configuration system for business users to customize the automotive appointment booking environment.
"""

import json
from pathlib import Path
from typing import Dict, Any

class AutomotiveConfig:
    """Easy configuration for automotive appointment booking environment."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize with default configuration."""
        self.config_file = config_file
        self.default_config = self._get_default_config()
        self.config = self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "environment": {
                "max_conversation_turns": 12,
                "num_customer_types": 6,
                "success_threshold": 0.6  # Minimum score to book appointment
            },
            "customer_psychology": {
                "interest": {
                    "description": "How interested the customer is in buying a car",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "ready_threshold": 0.5
                },
                "urgency": {
                    "description": "How soon the customer needs to buy",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "ready_threshold": 0.5
                },
                "availability": {
                    "description": "How flexible the customer's schedule is",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "ready_threshold": 0.4
                },
                "trust": {
                    "description": "How much the customer trusts the dealership",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "ready_threshold": 0.5
                },
                "commitment": {
                    "description": "How ready the customer is to take the next step",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "ready_threshold": 0.6
                }
            },
            "sales_actions": {
                "rapport": {
                    "description": "Build relationship and trust with customer",
                    "effects": {
                        "interest": 0.10,
                        "urgency": -0.02,
                        "availability": -0.01,
                        "trust": 0.06,
                        "commitment": 0.03
                    }
                },
                "qualify": {
                    "description": "Learn about customer needs and preferences",
                    "effects": {
                        "interest": 0.07,
                        "urgency": -0.03,
                        "availability": -0.01,
                        "trust": 0.05,
                        "commitment": 0.02
                    }
                },
                "show_inventory": {
                    "description": "Show available cars that match customer needs",
                    "effects": {
                        "interest": 0.12,
                        "urgency": 0.00,
                        "availability": -0.01,
                        "trust": 0.03,
                        "commitment": 0.04
                    }
                },
                "handle_concerns": {
                    "description": "Address customer worries about price, reliability, etc.",
                    "effects": {
                        "interest": 0.03,
                        "urgency": -0.10,
                        "availability": 0.00,
                        "trust": 0.02,
                        "commitment": 0.01
                    }
                },
                "create_urgency": {
                    "description": "Create time pressure with limited offers or popular models",
                    "effects": {
                        "interest": 0.04,
                        "urgency": -0.01,
                        "availability": -0.08,
                        "trust": 0.02,
                        "commitment": 0.05
                    }
                },
                "social_proof": {
                    "description": "Show that other customers have bought and are happy",
                    "effects": {
                        "interest": 0.02,
                        "urgency": 0.00,
                        "availability": 0.00,
                        "trust": 0.11,
                        "commitment": 0.03
                    }
                },
                "offer_incentives": {
                    "description": "Offer special deals, financing, or free services",
                    "effects": {
                        "interest": 0.06,
                        "urgency": -0.01,
                        "availability": -0.01,
                        "trust": 0.04,
                        "commitment": 0.08
                    }
                },
                "book_appointment": {
                    "description": "Try to book an appointment with the customer",
                    "effects": {
                        "interest": 0.00,
                        "urgency": 0.00,
                        "availability": 0.00,
                        "trust": 0.00,
                        "commitment": 0.00
                    }
                }
            },
            "customer_types": {
                "type_0": {
                    "name": "Cautious Buyer",
                    "description": "Takes time to decide, needs lots of information",
                    "base_psychology": {
                        "interest": 0.30,
                        "urgency": 0.20,
                        "availability": 0.40,
                        "trust": 0.25,
                        "commitment": 0.20
                    }
                },
                "type_1": {
                    "name": "Price Shopper",
                    "description": "Very focused on getting the best deal",
                    "base_psychology": {
                        "interest": 0.45,
                        "urgency": 0.35,
                        "availability": 0.60,
                        "trust": 0.35,
                        "commitment": 0.35
                    }
                },
                "type_2": {
                    "name": "Ready Buyer",
                    "description": "Already knows what they want, ready to buy",
                    "base_psychology": {
                        "interest": 0.60,
                        "urgency": 0.50,
                        "availability": 0.80,
                        "trust": 0.45,
                        "commitment": 0.50
                    }
                },
                "type_3": {
                    "name": "Research Buyer",
                    "description": "Wants to learn everything before deciding",
                    "base_psychology": {
                        "interest": 0.30,
                        "urgency": 0.20,
                        "availability": 0.40,
                        "trust": 0.25,
                        "commitment": 0.20
                    }
                },
                "type_4": {
                    "name": "Impulse Buyer",
                    "description": "Makes quick decisions, easy to convince",
                    "base_psychology": {
                        "interest": 0.45,
                        "urgency": 0.35,
                        "availability": 0.60,
                        "trust": 0.35,
                        "commitment": 0.35
                    }
                },
                "type_5": {
                    "name": "Skeptical Buyer",
                    "description": "Hard to convince, needs lots of proof",
                    "base_psychology": {
                        "interest": 0.60,
                        "urgency": 0.50,
                        "availability": 0.80,
                        "trust": 0.45,
                        "commitment": 0.50
                    }
                }
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {self.config_file}")
    
    def get_action_effects(self) -> list:
        """Get action effects matrix for the environment."""
        effects = []
        for action_name, action_config in self.config["sales_actions"].items():
            effect = action_config["effects"]
            effects.append([
                effect["interest"],
                effect["urgency"], 
                effect["availability"],
                effect["trust"],
                effect["commitment"]
            ])
        return effects
    
    def get_persona_priors(self) -> list:
        """Get persona priors for the environment."""
        priors = []
        for i in range(self.config["environment"]["num_customer_types"]):
            persona_key = f"type_{i}"
            if persona_key in self.config["customer_types"]:
                persona = self.config["customer_types"][persona_key]["base_psychology"]
                priors.append([
                    persona["interest"],
                    persona["urgency"],
                    persona["availability"],
                    persona["trust"],
                    persona["commitment"]
                ])
        return priors
    
    def update_action_effect(self, action_name: str, trait: str, value: float) -> None:
        """Update a specific action effect."""
        if action_name in self.config["sales_actions"]:
            self.config["sales_actions"][action_name]["effects"][trait] = value
            print(f"Updated {action_name} -> {trait}: {value}")
        else:
            print(f"Action '{action_name}' not found")
    
    def update_customer_type(self, type_id: int, trait: str, value: float) -> None:
        """Update a specific customer type trait."""
        type_key = f"type_{type_id}"
        if type_key in self.config["customer_types"]:
            self.config["customer_types"][type_key]["base_psychology"][trait] = value
            print(f"Updated {type_key} -> {trait}: {value}")
        else:
            print(f"Customer type {type_id} not found")
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("🔧 Current Configuration Summary:")
        print(f"📊 Max conversation turns: {self.config['environment']['max_conversation_turns']}")
        print(f"👥 Number of customer types: {self.config['environment']['num_customer_types']}")
        print(f"🎯 Success threshold: {self.config['environment']['success_threshold']}")
        
        print("\n🎭 Customer Types:")
        for type_key, type_config in self.config["customer_types"].items():
            print(f"  {type_config['name']}: {type_config['description']}")
        
        print("\n🎯 Sales Actions:")
        for action_name, action_config in self.config["sales_actions"].items():
            print(f"  {action_name}: {action_config['description']}")

def main():
    """Test the configuration system."""
    config = AutomotiveConfig()
    config.print_config_summary()
    
    # Example: Update an action effect
    print("\n🔧 Testing action effect update...")
    config.update_action_effect("rapport", "trust", 0.08)
    
    # Example: Update a customer type
    print("🔧 Testing customer type update...")
    config.update_customer_type(0, "interest", 0.35)
    
    # Save changes
    config.save_config()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Configuration Generator for RL Prompt Engine

This module allows users to generate configuration JSON files using natural language prompts,
making it easy to customize the RL environment without manually editing JSON files.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigGenerator:
    """Generates RL environment configuration using LLM prompts."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the configuration generator.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_config(self, 
                       description: str,
                       customer_types: Optional[List[str]] = None,
                       conversation_stages: Optional[List[str]] = None,
                       prompt_components: Optional[List[str]] = None,
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate configuration JSON from natural language description.
        
        Args:
            description: Natural language description of the business scenario
            customer_types: List of customer types (optional, will be generated if not provided)
            conversation_stages: List of conversation stages (optional, will be generated if not provided)
            prompt_components: List of prompt components (optional, will be generated if not provided)
            output_file: Path to save the generated config (optional)
            
        Returns:
            Generated configuration dictionary
        """
        print("🤖 Generating configuration from description...")
        
        # Create the prompt for the LLM
        prompt = self._create_config_prompt(
            description, customer_types, conversation_stages, prompt_components
        )
        
        # Generate configuration using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating RL environment configurations for appointment booking systems. Generate valid JSON configurations based on user descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        # Parse the response
        config_text = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown formatting if present)
        if config_text.startswith("```json"):
            config_text = config_text[7:]
        if config_text.endswith("```"):
            config_text = config_text[:-3]
        
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing generated JSON: {e}")
            print("Generated text:")
            print(config_text)
            raise ValueError("Failed to parse generated configuration JSON")
        
        # Validate the configuration
        self._validate_config(config)
        
        # Save to file if specified
        if output_file:
            self._save_config(config, output_file)
        
        print("✅ Configuration generated successfully!")
        return config
    
    def _create_config_prompt(self, 
                             description: str,
                             customer_types: Optional[List[str]] = None,
                             conversation_stages: Optional[List[str]] = None,
                             prompt_components: Optional[List[str]] = None) -> str:
        """Create the prompt for generating configuration."""
        
        prompt = f"""
Generate a comprehensive RL environment configuration for an appointment booking system based on this description:

DESCRIPTION: {description}

The configuration should include:

1. CUSTOMER TYPES (6 types):
{f"- {', '.join(customer_types)}" if customer_types else "- Generate 6 different customer personas relevant to the business scenario"}

2. CONVERSATION STAGES (4 stages):
{f"- {', '.join(conversation_stages)}" if conversation_stages else "- Generate 4 stages of the sales conversation process"}

3. PROMPT COMPONENTS (10 components):
{f"- {', '.join(prompt_components)}" if prompt_components else "- Generate 10 different prompt components for appointment booking"}

4. URGENCY LEVELS: low, medium, high

5. CUSTOMER PSYCHOLOGY DIMENSIONS: interest, urgency, availability, trust, commitment

The configuration should be realistic, business-appropriate, and suitable for training an RL agent to generate effective appointment booking prompts.

Generate a complete JSON configuration with the following structure:
{{
    "customer_types": {{
        "type_name": {{
            "description": "Description of this customer type",
            "characteristics": ["trait1", "trait2", "trait3"],
            "approach": "How to approach this customer type",
            "tone": "Recommended tone for this customer type"
        }}
    }},
    "conversation_stages": {{
        "stage_name": {{
            "description": "Description of this conversation stage",
            "goals": ["goal1", "goal2"],
            "approach": "How to handle this stage"
        }}
    }},
    "prompt_components": {{
        "component_name": {{
            "description": "Description of this component",
            "template": "Template text with placeholders like {{parameter}}",
            "use_case": "When to use this component",
            "effectiveness": 0.8
        }}
    }},
    "urgency_levels": {{
        "low": {{
            "description": "Low urgency scenario",
            "time_reference": "when convenient",
            "approach": "Patient, no pressure"
        }},
        "medium": {{
            "description": "Medium urgency scenario", 
            "time_reference": "this week",
            "approach": "Moderate pressure, create some urgency"
        }},
        "high": {{
            "description": "High urgency scenario",
            "time_reference": "today or tomorrow", 
            "approach": "Strong urgency, immediate action needed"
        }}
    }},
    "customer_psychology": {{
        "dimensions": ["interest", "urgency", "availability", "trust", "commitment"],
        "descriptions": {{
            "interest": "Customer's interest level in the product/service",
            "urgency": "Customer's urgency to make a decision",
            "availability": "Customer's availability for appointments",
            "trust": "Customer's trust level in the salesperson/company",
            "commitment": "Customer's commitment level to purchasing"
        }}
    }},
    "business_context": {{
        "industry": "Industry type",
        "product_type": "Type of product/service",
        "appointment_duration": "Typical appointment length",
        "sales_cycle": "Typical sales cycle length"
    }}
}}

Make sure the configuration is specific to the business scenario described and includes realistic, actionable content.
"""
        return prompt
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the generated configuration."""
        required_sections = ["customer_types", "conversation_stages", "prompt_components", "urgency_levels", "customer_psychology"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate customer types
        if len(config["customer_types"]) != 6:
            raise ValueError(f"Expected 6 customer types, got {len(config['customer_types'])}")
        
        # Validate conversation stages
        if len(config["conversation_stages"]) != 4:
            raise ValueError(f"Expected 4 conversation stages, got {len(config['conversation_stages'])}")
        
        # Validate prompt components
        if len(config["prompt_components"]) != 10:
            raise ValueError(f"Expected 10 prompt components, got {len(config['prompt_components'])}")
        
        print("✅ Configuration validation passed!")
    
    def _save_config(self, config: Dict[str, Any], output_file: str) -> None:
        """Save configuration to file."""
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {output_file}")
    
    def generate_from_examples(self, examples: List[Dict[str, str]], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate configuration from example scenarios.
        
        Args:
            examples: List of example scenarios with descriptions
            output_file: Path to save the generated config (optional)
            
        Returns:
            Generated configuration dictionary
        """
        examples_text = "\n".join([f"- {ex['name']}: {ex['description']}" for ex in examples])
        
        prompt = f"""
Based on these example scenarios, generate a comprehensive RL environment configuration:

EXAMPLES:
{examples_text}

Generate a configuration that can handle all these scenarios and similar ones.
"""
        
        return self.generate_config(prompt, output_file=output_file)

def create_config_from_prompt(description: str, output_file: str = "configs/generated_config.json") -> Dict[str, Any]:
    """
    Convenience function to create configuration from a prompt.
    
    Args:
        description: Natural language description of the business scenario
        output_file: Path to save the generated config
        
    Returns:
        Generated configuration dictionary
    """
    generator = ConfigGenerator()
    return generator.generate_config(description, output_file=output_file)

def create_config_from_examples(examples: List[Dict[str, str]], output_file: str = "configs/generated_config.json") -> Dict[str, Any]:
    """
    Convenience function to create configuration from examples.
    
    Args:
        examples: List of example scenarios
        output_file: Path to save the generated config
        
    Returns:
        Generated configuration dictionary
    """
    generator = ConfigGenerator()
    return generator.generate_from_examples(examples, output_file=output_file)

if __name__ == "__main__":
    # Example usage
    examples = [
        {
            "name": "Car Dealership",
            "description": "A car dealership selling new and used vehicles, with customers ranging from first-time buyers to luxury car enthusiasts"
        },
        {
            "name": "Real Estate Agency", 
            "description": "A real estate agency helping clients buy and sell homes, with customers at different stages of the home buying process"
        },
        {
            "name": "SaaS Company",
            "description": "A software company selling subscription-based business tools to small and medium enterprises"
        }
    ]
    
    generator = ConfigGenerator()
    config = generator.generate_from_examples(examples, "multi_industry_config.json")
    print("Configuration generated successfully!")

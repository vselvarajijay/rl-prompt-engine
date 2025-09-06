#!/usr/bin/env python3
"""
Generic Prompt Generator

Generates prompt templates from RL agent decisions using configuration files.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Represents a generated prompt template."""
    template: str
    components: List[str]
    context_type: str
    conversation_stage: str
    urgency_level: str
    effectiveness_score: float
    metadata: Dict[str, Any]

class PromptGenerator:
    """Generates prompt templates from RL agent decisions."""
    
    def __init__(self, config_file: str = "configs/default_config.json"):
        """Initialize the prompt generator with configuration."""
        self.config = self._load_config(config_file)
        self.prompt_components = self.config["prompt_components"]
        self.context_types = self.config["context_types"]
        self.conversation_stages = self.config["conversation_stages"]
        self.urgency_levels = self.config["urgency_levels"]
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_template(self, 
                         context_type: int, 
                         conversation_stage: int,
                         urgency_level: int,
                         selected_components: List[int],
                         custom_variables: Optional[Dict[str, str]] = None) -> PromptTemplate:
        """
        Generate a prompt template from selected components.
        
        Args:
            context_type: Index of context type
            conversation_stage: Index of conversation stage
            urgency_level: Index of urgency level
            selected_components: List of selected component indices
            custom_variables: Optional custom variables to fill in templates
            
        Returns:
            Generated PromptTemplate object
        """
        # Get names from indices
        context_type_name = list(self.context_types.keys())[context_type]
        stage_name = list(self.conversation_stages.keys())[conversation_stage]
        urgency_name = list(self.urgency_levels.keys())[urgency_level]
        
        # Get component names
        selected_component_names = [list(self.prompt_components.keys())[int(i)] for i in selected_components]
        
        # Generate the actual prompt template
        template_parts = []
        
        for component_name in selected_component_names:
            if component_name in self.prompt_components:
                component_config = self.prompt_components[component_name]
                template_part = component_config.get("template", f"[{component_name}]")
                
                # Fill in default variables if custom_variables not provided
                if not custom_variables:
                    custom_variables = self._get_default_variables(
                        context_type_name, stage_name, urgency_name
                    )
                
                # Fill in variables
                for key, value in custom_variables.items():
                    template_part = template_part.replace(f"{{{key}}}", value)
                
                template_parts.append(template_part)
        
        # Combine all parts
        full_template = "\n\n".join(template_parts)
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness(
            selected_component_names, context_type_name, stage_name, urgency_name
        )
        
        # Create metadata
        metadata = {
            "component_count": len(selected_components),
            "generation_timestamp": np.datetime64('now').astype(str),
            "config_version": "2.0"
        }
        
        return PromptTemplate(
            template=full_template,
            components=selected_component_names,
            context_type=context_type_name,
            conversation_stage=stage_name,
            urgency_level=urgency_name,
            effectiveness_score=effectiveness_score,
            metadata=metadata
        )
    
    def _get_default_variables(self, context_type: str, conversation_stage: str, urgency_level: str) -> Dict[str, str]:
        """Get default variables for template filling."""
        return {
            "context_type": context_type,
            "conversation_stage": conversation_stage,
            "urgency_level": urgency_level,
            "first_name": "Customer",
            "product": "our product",
            "company_name": "our company",
            "budget": "your budget",
            "timeline": "your timeline",
            "benefit": "key benefit",
            "rate": "special rate",
            "incentive": "special offer",
            "concern": "your concern",
            "time_reference": "when convenient"
        }
    
    def _calculate_effectiveness(self, component_names: List[str], context_type_name: str, 
                               stage_name: str, urgency_name: str) -> float:
        """Calculate effectiveness score for selected components."""
        if not component_names:
            return 0.0
        
        total_effectiveness = 0.0
        
        for component_name in component_names:
            component_config = self.prompt_components[component_name]
            base_effectiveness = component_config.get("effectiveness", 0.5)
            
            # Context type bonus
            context_config = self.context_types[context_type_name]
            if component_name in context_config.get("preferred_components", []):
                base_effectiveness *= 1.2
            
            # Stage bonus
            stage_config = self.conversation_stages[stage_name]
            if component_name in stage_config.get("preferred_components", []):
                base_effectiveness *= 1.1
            
            # Urgency bonus
            urgency_config = self.urgency_levels[urgency_name]
            if urgency_name == "high" and component_name in urgency_config.get("preferred_components", []):
                base_effectiveness *= 1.15
            
            total_effectiveness += base_effectiveness
        
        return total_effectiveness / len(component_names)
    
    def generate_meta_prompt(self, template: PromptTemplate) -> str:
        """Generate a meta-prompt template for use with LLMs."""
        context_config = self.context_types[template.context_type]
        stage_config = self.conversation_stages[template.conversation_stage]
        urgency_config = self.urgency_levels[template.urgency_level]
        
        meta_prompt = f"""You are an AI assistant for {template.context_type} customers. Generate a message with the following specifications:

CUSTOMER PROFILE:
- Context Type: {template.context_type}
- Conversation Stage: {template.conversation_stage}
- Urgency Level: {template.urgency_level}
- Description: {context_config.get('description', 'Customer needs specialized approach')}

TONE AND APPROACH:
- Tone: {context_config.get('tone', 'professional and helpful')}
- Approach: {context_config.get('approach', 'adapt to customer needs')}
- Time Reference: {urgency_config.get('time_reference', 'when convenient')}

MESSAGE TEMPLATE:
{template.template}

PARAMETERS TO FILL:
- first_name: Customer's first name
- product: Specific product/service they're interested in
- company_name: Name of your company
- budget: Customer's budget range
- timeline: When they need the solution
- benefit: Main benefit of the product/service
- rate: Special rate or offer
- incentive: Special offer or incentive
- concern: Specific concern to address
- time_reference: When to schedule (e.g., today, this week, when convenient)

INSTRUCTIONS:
1. Fill in all parameters with appropriate values
2. Use the {context_config.get('tone', 'professional')} tone throughout
3. Incorporate all template parts in a natural flow
4. Keep the message conversational and professional
5. End with a clear call-to-action
6. Make it sound like a real conversation

Generate a complete message that follows this template and incorporates all specified elements."""
        
        return meta_prompt

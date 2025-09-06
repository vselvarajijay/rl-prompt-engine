#!/usr/bin/env python3
"""
Appointment Booking Prompt Generator and Template System

This module provides tools to:
1. Generate actual appointment booking prompts from RL agent decisions
2. Create templates for different customer types and conversation stages
3. Evaluate prompt effectiveness
4. Create embeddings for prompt indexing and retrieval
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import hashlib
from dataclasses import dataclass
from .appointment_prompt_env import AppointmentPromptEnv

@dataclass
class AppointmentPromptTemplate:
    """Represents a generated appointment booking prompt template."""
    template: str
    components: List[str]
    customer_type: str
    conversation_stage: str
    urgency_level: str
    effectiveness_score: float
    template_id: str
    metadata: Dict[str, Any]

class AppointmentPromptGenerator:
    """Generates actual appointment booking prompts from RL agent decisions."""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.component_templates = self._load_component_templates()
        self.customer_types = list(self.config["customer_types"].keys())
        self.conversation_stages = list(self.config["conversation_stages"].keys())
        self.urgency_levels = ["low", "medium", "high"]
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration file."""
        if config_file is None:
            # Use the package config file
            config_path = Path(__file__).parent.parent / "config" / "appointment_prompt_config.json"
        else:
            config_path = Path(config_file)
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_component_templates(self) -> Dict[str, str]:
        """Load actual prompt templates for each component."""
        return {
            "rapport_building": """Hi {customer_name}! I hope you're having a great {time_of_day}. I'm {agent_name} from {company_name}, and I wanted to personally reach out about your interest in our {product_type}. 

I noticed you've been looking at our {specific_product} - that's a fantastic choice! Many of our customers have been thrilled with it.""",

            "needs_assessment": """To make sure I can help you find exactly what you're looking for, could you tell me:

• What's most important to you in a {product_type}?
• Are you looking for something specific for {use_case}?
• What's your timeline for making a decision?
• Do you have any particular features or requirements in mind?

This will help me show you the best options we have available.""",

            "value_proposition": """Here's what makes our {product_type} special:

✨ {key_benefit_1}
✨ {key_benefit_2}  
✨ {key_benefit_3}

Plus, we're currently offering {special_offer} which could save you {savings_amount}. This is exactly the kind of value our customers love about working with us.""",

            "objection_handling": """I completely understand your concern about {objection}. That's actually one of the most common questions I get, and here's what I tell people:

{objection_response}

The great news is that {positive_reassurance}. Would you like me to show you some examples of how this has worked for other customers like you?""",

            "urgency_creation": """I wanted to let you know that we only have {limited_quantity} of these {product_type} left in stock, and they're going fast. 

In fact, just this week we've had {recent_sales} customers come in for the same model. The {special_feature} feature you mentioned being interested in is particularly popular right now.

I'd hate for you to miss out on this opportunity, especially with our current {promotion_name} promotion.""",

            "social_proof": """You know, I was just thinking - last month alone, we helped {number_of_customers} customers find their perfect {product_type}. 

One customer, {customer_example_name}, came in with similar needs to yours and said, "{customer_testimonial}"

It's stories like that that make me love what I do. When customers are genuinely happy with their choice, it shows we're doing something right.""",

            "incentive_offering": """I have some great news! Right now we're running a special promotion that I think you'll really appreciate:

🎁 {incentive_1}
🎁 {incentive_2}
🎁 {incentive_3}

This promotion is only available until {promotion_end_date}, and I'd love to make sure you can take advantage of it. The savings alone could be worth {total_savings}.""",

            "appointment_booking": """I'd love to show you our {product_type} in person so you can see exactly what I'm talking about. 

Are you free {suggested_time_1} or {suggested_time_2}? I can set aside some time to give you a personalized tour and answer any questions you might have.

I'll even make sure to have {specific_preparation} ready for you when you come in.""",

            "follow_up": """I'll make sure to follow up with you {follow_up_timeline} to see if you have any other questions.

In the meantime, I'll put together some additional information about {relevant_topic} that I think you'll find helpful.

Is {contact_method} the best way to reach you, or would you prefer {alternative_contact}?""",

            "personalization": """Based on what you've told me about {customer_specific_need}, I think you'd be particularly interested in {personalized_recommendation}.

I've actually helped several customers in {customer_situation} find exactly what they needed, and they've all been thrilled with the results.

Let me show you why this might be perfect for your specific situation..."""
        }
    
    def generate_prompt_template(self, 
                               customer_type: int, 
                               conversation_stage: int,
                               urgency_level: int,
                               selected_components: List[int],
                               customer_psychology: np.ndarray,
                               custom_variables: Dict[str, str] = None) -> AppointmentPromptTemplate:
        """Generate a complete appointment booking prompt template from RL agent decisions."""
        
        # Get context names
        customer_type_name = self.customer_types[customer_type]
        conversation_stage_name = self.conversation_stages[conversation_stage]
        urgency_level_name = self.urgency_levels[urgency_level]
        
        # Get component names
        component_names = list(self.config["prompt_components"].keys())
        selected_component_names = [component_names[i] for i in selected_components]
        
        # Generate the actual prompt template
        template_parts = []
        
        for component_name in selected_component_names:
            if component_name in self.component_templates:
                template_part = self.component_templates[component_name]
                
                # Fill in default variables if custom_variables not provided
                if not custom_variables:
                    custom_variables = self._get_default_variables(
                        customer_type_name, conversation_stage_name, urgency_level_name
                    )
                
                # Fill in variables
                for key, value in custom_variables.items():
                    template_part = template_part.replace(f"{{{key}}}", value)
                
                template_parts.append(template_part)
        
        # Combine all parts
        full_template = "\n\n".join(template_parts)
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness(
            selected_component_names, customer_type_name, conversation_stage_name, urgency_level_name
        )
        
        # Generate unique template ID
        template_id = self._generate_template_id(
            customer_type_name, conversation_stage_name, urgency_level_name, selected_component_names
        )
        
        # Create metadata
        metadata = {
            "customer_psychology": customer_psychology.tolist(),
            "component_count": len(selected_components),
            "generation_timestamp": np.datetime64('now').astype(str),
            "config_version": "1.0"
        }
        
        return AppointmentPromptTemplate(
            template=full_template,
            components=selected_component_names,
            customer_type=customer_type_name,
            conversation_stage=conversation_stage_name,
            urgency_level=urgency_level_name,
            effectiveness_score=effectiveness_score,
            template_id=template_id,
            metadata=metadata
        )
    
    def _get_default_variables(self, customer_type: str, conversation_stage: str, urgency_level: str) -> Dict[str, str]:
        """Get default variables for template filling."""
        return {
            "customer_name": "there",
            "time_of_day": "day",
            "agent_name": "Sarah",
            "company_name": "AutoMax Dealership",
            "product_type": "vehicle",
            "specific_product": "2024 Honda Civic",
            "use_case": "your daily commute",
            "key_benefit_1": "Outstanding fuel efficiency - up to 42 MPG highway",
            "key_benefit_2": "Advanced safety features including Honda Sensing",
            "key_benefit_3": "5-year warranty with roadside assistance",
            "special_offer": "0.9% APR financing",
            "savings_amount": "thousands of dollars",
            "objection": "the price",
            "objection_response": "I completely understand wanting to get the best value. Let me show you how our financing options can make this more affordable than you might think.",
            "positive_reassurance": "our financing team has helped hundreds of customers find payment plans that work for their budget",
            "limited_quantity": "3",
            "recent_sales": "12",
            "special_feature": "fuel-efficient",
            "promotion_name": "Spring Special",
            "number_of_customers": "47",
            "customer_example_name": "Mike",
            "customer_testimonial": "I couldn't be happier with my decision. The fuel savings alone have paid for the difference in price.",
            "incentive_1": "Free extended warranty (valued at $2,500)",
            "incentive_2": "Complimentary maintenance for 2 years",
            "incentive_3": "Free installation of premium floor mats",
            "promotion_end_date": "this Friday",
            "total_savings": "$3,200",
            "suggested_time_1": "tomorrow afternoon around 2 PM",
            "suggested_time_2": "Thursday morning at 10 AM",
            "specific_preparation": "the exact model you're interested in",
            "follow_up_timeline": "in a couple of days",
            "relevant_topic": "financing options and warranty details",
            "contact_method": "phone",
            "alternative_contact": "email",
            "customer_specific_need": "reliable transportation for your family",
            "personalized_recommendation": "our family-friendly SUV with all-wheel drive",
            "customer_situation": "similar situations"
        }
    
    def _calculate_effectiveness(self, 
                               component_names: List[str], 
                               customer_type: str, 
                               conversation_stage: str, 
                               urgency_level: str) -> float:
        """Calculate effectiveness score for the prompt template."""
        total_score = 0.0
        
        for component_name in component_names:
            if component_name in self.config["prompt_components"]:
                component_config = self.config["prompt_components"][component_name]
                base_score = component_config["effectiveness"][customer_type]
                stage_score = component_config["stage_effectiveness"][conversation_stage]
                combined_score = (base_score + stage_score) / 2
                
                # Customer preference bonus
                customer_preferences = self.config["customer_types"][customer_type]["preferences"]
                if component_name in customer_preferences:
                    combined_score *= 1.2
                
                # Stage preference bonus
                stage_preferences = self.config["conversation_stages"][conversation_stage]["preferred_components"]
                if component_name in stage_preferences:
                    combined_score *= 1.1
                
                # Urgency bonus
                if urgency_level == "high" and component_name in ["urgency_creation", "appointment_booking"]:
                    combined_score *= 1.15
                
                total_score += combined_score
        
        return min(total_score / len(component_names), 1.0) if component_names else 0.0
    
    def _generate_template_id(self, 
                            customer_type: str, 
                            conversation_stage: str, 
                            urgency_level: str, 
                            components: List[str]) -> str:
        """Generate a unique ID for the template."""
        content = f"{customer_type}_{conversation_stage}_{urgency_level}_{'_'.join(sorted(components))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def create_embedding(self, template: AppointmentPromptTemplate) -> np.ndarray:
        """Create a simple embedding for the template."""
        # Simple bag-of-words style embedding
        all_components = list(self.config["prompt_components"].keys())
        embedding = np.zeros(len(all_components) + 6)  # +6 for context features
        
        # Component presence
        for i, component in enumerate(all_components):
            if component in template.components:
                embedding[i] = 1.0
        
        # Context features
        embedding[len(all_components)] = self.customer_types.index(template.customer_type) / len(self.customer_types)
        embedding[len(all_components) + 1] = self.conversation_stages.index(template.conversation_stage) / len(self.conversation_stages)
        embedding[len(all_components) + 2] = self.urgency_levels.index(template.urgency_level) / len(self.urgency_levels)
        embedding[len(all_components) + 3] = template.effectiveness_score
        embedding[len(all_components) + 4] = len(template.components) / 10.0  # Normalized component count
        embedding[len(all_components) + 5] = 1.0 if "appointment_booking" in template.components else 0.0  # Has booking component
        
        return embedding

class AppointmentPromptDatabase:
    """Database for storing and retrieving appointment booking prompt templates."""
    
    def __init__(self, db_file: str = "appointment_prompt_database.json"):
        self.db_file = db_file
        self.templates = self._load_database()
    
    def _load_database(self) -> Dict[str, AppointmentPromptTemplate]:
        """Load templates from database file."""
        if Path(self.db_file).exists():
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                return {k: AppointmentPromptTemplate(**v) for k, v in data.items()}
        return {}
    
    def save_database(self):
        """Save templates to database file."""
        data = {k: {
            'template': v.template,
            'components': v.components,
            'customer_type': v.customer_type,
            'conversation_stage': v.conversation_stage,
            'urgency_level': v.urgency_level,
            'effectiveness_score': v.effectiveness_score,
            'template_id': v.template_id,
            'metadata': v.metadata
        } for k, v in self.templates.items()}
        
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_template(self, template: AppointmentPromptTemplate):
        """Add a template to the database."""
        self.templates[template.template_id] = template
        self.save_database()
    
    def search_templates(self, 
                        customer_type: str = None,
                        conversation_stage: str = None,
                        urgency_level: str = None,
                        min_effectiveness: float = 0.0) -> List[AppointmentPromptTemplate]:
        """Search for templates matching criteria."""
        results = []
        
        for template in self.templates.values():
            if customer_type and template.customer_type != customer_type:
                continue
            if conversation_stage and template.conversation_stage != conversation_stage:
                continue
            if urgency_level and template.urgency_level != urgency_level:
                continue
            if template.effectiveness_score < min_effectiveness:
                continue
            
            results.append(template)
        
        return sorted(results, key=lambda x: x.effectiveness_score, reverse=True)
    
    def get_best_template(self, 
                         customer_type: str, 
                         conversation_stage: str, 
                         urgency_level: str) -> Optional[AppointmentPromptTemplate]:
        """Get the best template for given criteria."""
        candidates = self.search_templates(customer_type, conversation_stage, urgency_level)
        return candidates[0] if candidates else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.templates:
            return {"total_templates": 0}
        
        total = len(self.templates)
        effectiveness_scores = [t.effectiveness_score for t in self.templates.values()]
        
        return {
            "total_templates": total,
            "avg_effectiveness": np.mean(effectiveness_scores),
            "max_effectiveness": np.max(effectiveness_scores),
            "min_effectiveness": np.min(effectiveness_scores),
            "customer_type_distribution": {
                ct: sum(1 for t in self.templates.values() if t.customer_type == ct)
                for ct in set(t.customer_type for t in self.templates.values())
            },
            "conversation_stage_distribution": {
                cs: sum(1 for t in self.templates.values() if t.conversation_stage == cs)
                for cs in set(t.conversation_stage for t in self.templates.values())
            }
        }

def main():
    """Demo the appointment prompt generation system."""
    print("🎯 Appointment Booking Meta-Prompting System Demo")
    print("=" * 50)
    
    # Initialize components
    generator = AppointmentPromptGenerator()
    database = AppointmentPromptDatabase()
    
    # Create some example templates
    print("📝 Generating example appointment booking prompts...")
    
    # Example 1: Cautious customer, early stage, low urgency
    template1 = generator.generate_prompt_template(
        customer_type=0,  # cautious
        conversation_stage=0,  # early
        urgency_level=0,  # low
        selected_components=[0, 1, 9],  # rapport_building, needs_assessment, personalization
        customer_psychology=np.array([0.3, 0.2, 0.4, 0.25, 0.2])
    )
    
    print(f"\n📋 Template 1 (Cautious Customer, Early Stage):")
    print(f"Effectiveness: {template1.effectiveness_score:.3f}")
    print(f"Components: {', '.join(template1.components)}")
    print(f"Template Preview: {template1.template[:200]}...")
    
    # Example 2: Ready buyer, closing stage, high urgency
    template2 = generator.generate_prompt_template(
        customer_type=2,  # ready_buyer
        conversation_stage=3,  # closing
        urgency_level=2,  # high
        selected_components=[2, 4, 7],  # value_proposition, urgency_creation, appointment_booking
        customer_psychology=np.array([0.6, 0.5, 0.8, 0.45, 0.5])
    )
    
    print(f"\n📋 Template 2 (Ready Buyer, Closing Stage):")
    print(f"Effectiveness: {template2.effectiveness_score:.3f}")
    print(f"Components: {', '.join(template2.components)}")
    print(f"Template Preview: {template2.template[:200]}...")
    
    # Add to database
    database.add_template(template1)
    database.add_template(template2)
    
    print(f"\n📊 Database Statistics:")
    stats = database.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Demo complete! The system is ready to generate appointment booking prompts.")

if __name__ == "__main__":
    main()

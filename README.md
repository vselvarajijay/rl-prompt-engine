# 🤖 RL Prompt Engine

A **generic, configurable** reinforcement learning-powered system for generating dynamic prompt templates for **any use case**. The system uses PPO (Proximal Policy Optimization) to learn optimal prompt strategies based on context types, conversation stages, and urgency levels.

## 🎯 What This Does

This AI learns to generate optimal prompts for any business conversation by understanding context and adapting its approach. It learns through trial and error, getting better over time at creating effective prompts for different scenarios.

## 🚀 Quick Start

### 1. Installation
```bash
cd rl_prompt_engine
poetry install
```

### 2. Train a Model
```bash
# Train with default configuration
poetry run python -m rl_prompt_engine.cli_simple train --timesteps 2000 --save-path models/demo_model
```

### 3. Generate Prompts
```bash
# Generate a prompt for new customers
poetry run python -m rl_prompt_engine.cli_simple generate \
  --model-path models/demo_model \
  --context-type 0 \
  --conversation-stage 0 \
  --urgency-level 0 \
  --output new_customer_greeting.txt
```

### 4. Use with Any LLM
Copy the generated prompt and use it with ChatGPT, Claude, or any other LLM to create personalized messages.

## 🎭 What You'll See

The AI will generate **context-aware prompts** like this:

```
🎯 Context: New Customer, Opening Stage, Low Urgency

📝 Generated Meta-Prompt:
You are an AI assistant for new_customer customers. Generate a message with the following specifications:

CUSTOMER PROFILE:
- Context Type: new_customer
- Conversation Stage: opening
- Urgency Level: low
- Description: A new customer who has never used our service before

TONE AND APPROACH:
- Tone: Friendly, patient, reassuring
- Approach: Educate about the product, provide guidance and assurance
- Time Reference: when convenient

MESSAGE TEMPLATE:
We're offering special offer for when convenient

PARAMETERS TO FILL:
- first_name: Customer's first name
- product: Specific product/service they're interested in
- company_name: Name of your company
- benefit: Main benefit of the product/service

INSTRUCTIONS:
1. Fill in all parameters with appropriate values
2. Use the Friendly, patient, reassuring tone throughout
3. Incorporate all template parts in a natural flow
4. Keep the message conversational and professional
5. End with a clear call-to-action

Generate a complete message that follows this template and incorporates all specified elements.
```

## ⚙️ Easy Configuration

You can customize the AI's behavior by editing JSON configuration files. No coding required!

### Context Types

Define different user personas:

```json
{
  "context_types": {
    "new_customer": {
      "description": "A new customer who has never used our service before",
      "tone": "Friendly, patient, reassuring",
      "approach": "Educate about the product, provide guidance and assurance",
      "preferred_components": ["greeting", "needs_assessment", "reassurance"]
    },
    "premium_customer": {
      "description": "A customer who values quality and is less price-sensitive",
      "tone": "Professional, sophisticated, exclusive",
      "approach": "Emphasize quality, exclusivity, and premium features",
      "preferred_components": ["value_proposition", "social_proof", "call_to_action"]
    }
  }
}
```

### Prompt Components

Customize how each prompt component works:

```json
{
  "prompt_components": {
    "greeting": {
      "description": "Initial greeting to the customer",
      "template": "Hello {first_name}, how can we assist you today?",
      "effectiveness": 0.9
    },
    "value_proposition": {
      "description": "Presenting the value proposition",
      "template": "By choosing {product}, you'll benefit from {benefit}",
      "effectiveness": 0.8
    }
  }
}
```

### Conversation Stages

Define different stages of interaction:

```json
{
  "conversation_stages": {
    "opening": {
      "description": "Initial contact with the customer",
      "goals": ["Establish rapport", "Identify customer type"],
      "preferred_components": ["greeting", "needs_assessment"]
    },
    "closing": {
      "description": "Finalizing the deal",
      "goals": ["Close the deal", "Ensure satisfaction"],
      "preferred_components": ["call_to_action", "urgency_creation"]
    }
  }
}
```

## 🎭 Prompt Components Explained

| Component | What It Does | Best For |
|--------|-------------|----------|
| **Greeting** | Establishes initial connection | Starting conversations |
| **Needs Assessment** | Learns about user requirements | Understanding what they want |
| **Value Proposition** | Presents key benefits | When they're interested |
| **Handle Concerns** | Addresses objections and worries | When they're hesitant |
| **Urgency Creation** | Creates time pressure | When they're ready but slow |
| **Social Proof** | Shows others' success stories | Building confidence |
| **Incentive Offering** | Presents special offers | Closing the deal |
| **Call to Action** | Encourages next steps | When they're ready |

## 📊 Understanding Results

### Training Results
- **Success Rate**: Percentage of effective prompts generated
- **Component Distribution**: Which prompt components the AI uses most often

### Example Output
```
Success rate over 500 episodes: 0.982

Component distribution:
  Component 0:  366 (  8.2%)  # Greeting
  Component 1:  298 (  6.7%)  # Needs Assessment  
  Component 2:  466 ( 10.4%)  # Value Proposition
  Component 3:  365 (  8.2%)  # Handle Concerns
  Component 5:  798 ( 17.9%)  # Social Proof
  Component 6: 1833 ( 41.1%)  # Incentive Offering
  Component 7:  334 (  7.5%)  # Call to Action
```

## 🔧 Advanced Configuration

### Environment Settings
```json
{
  "environment": {
    "max_conversation_turns": 12,
    "num_customer_types": 6,
    "success_threshold": 0.6
  }
}
```

### Customizing Action Effects

To make an action more effective, increase its values:

```json
{
  "sales_actions": {
    "offer_incentives": {
      "effects": {
        "commitment": 0.12  # Increased from 0.08
      }
    }
  }
}
```

### Adding New Customer Types

```json
{
  "customer_types": {
    "type_6": {
      "name": "Luxury Buyer",
      "description": "High-end customer, price not a concern",
      "base_psychology": {
        "interest": 0.70,
        "urgency": 0.30,
        "availability": 0.90,
        "trust": 0.60,
        "commitment": 0.50
      }
    }
  }
}
```

## 🎯 Business Use Cases

### 1. **Sales & Marketing**
- Generate personalized sales messages
- Create targeted email campaigns
- Optimize lead generation prompts

### 2. **Customer Support**
- Generate context-aware support responses
- Create escalation prompts
- Improve customer satisfaction

### 3. **Content Creation**
- Generate marketing copy
- Create social media content
- Develop educational materials

### 4. **User Onboarding**
- Generate welcome sequences
- Create tutorial prompts
- Improve user engagement

## 📈 Performance Tips

### High Success Rate (>90%)
- Your AI is well-trained
- Consider making customers more challenging
- Try different customer type distributions

### Low Success Rate (<70%)
- Increase training time
- Check action effects are realistic
- Verify customer thresholds make sense

### Unbalanced Action Usage
- One action used too much: Reduce its effectiveness
- One action never used: Increase its effectiveness
- All actions used equally: Add more variety to customer types

## 🛠️ Troubleshooting

### Common Issues

**AI always uses the same action:**
- Check if one action is too effective
- Reduce its values in config.json

**AI never books appointments:**
- Lower the success thresholds
- Increase commitment values for actions

**Training is too slow:**
- Reduce max_conversation_turns
- Use fewer customer types initially

### Getting Help

1. Check the configuration file is valid JSON
2. Verify all required fields are present
3. Test with a simple configuration first
4. Check the logs for error messages

## 📁 File Structure

```
rl_prompt_engine/
├── rl_prompt_engine/           # Main package
│   ├── cli_simple.py          # Simplified command line interface
│   ├── core/                   # Core functionality
│   │   ├── prompt_system.py   # Main RL system class
│   │   ├── prompt_env.py      # Generic RL environment
│   │   ├── prompt_generator.py # Template generation
│   │   ├── config_generator.py # AI-powered config generation
│   │   ├── logging_config.py   # Logging setup
│   │   └── training_callback.py # Training callbacks
│   └── config/                 # Default configurations
├── configs/                    # Configuration files
│   ├── generic_config.json    # Generic configuration
│   ├── luxury_config.json     # Luxury car sales example
│   └── my_custom_config.json  # Your custom config
├── models/                     # Trained RL models
└── README.md                   # This file
```

## 🔧 Technical Notes

### Environment Details
- **Environment**: `PromptEnv` with discrete actions (prompt components) and context features
- **Reward**: Sparse reward - only at episode end (1 if effective prompt generated, 0 otherwise)
- **Actions**: 8 different prompt components (greeting, needs_assessment, value_proposition, etc.)
- **Context Features**: 5 contextual traits (context_type, conversation_stage, urgency_level, component_history, effectiveness)

### Customization
- Tune `component_effectiveness` in config files to shape task difficulty
- Adjust context type thresholds for different markets
- Modify conversation stage preferences for different use cases

## 🎉 Success!

Your RL Prompt Engine is now ready to generate optimal prompts for any business scenario! The system will automatically adapt to your configuration and get better over time.

Remember: The AI learns through trial and error, so the more you train it, the better it gets at generating effective prompts!
# 🤖 RL Prompt Engine

A **generic, configurable** reinforcement learning-powered system for generating dynamic prompt templates for **any use case**. The system uses PPO (Proximal Policy Optimization) to learn optimal prompt strategies based on context types, conversation stages, and urgency levels.

## ✨ What's New in v2.0

- 🎯 **Fully Generic**: Works for any business use case (sales, support, marketing, etc.)
- 🏗️ **Simplified Architecture**: 3 core classes instead of 8
- ⚙️ **Config-Driven**: Everything controlled by JSON configuration files
- 🚀 **Easy to Use**: Simple CLI and Python API
- 🔧 **No Hardcoded Logic**: Completely customizable via config
- 🤖 **RL-Powered**: Uses PPO to learn optimal prompt strategies

## 🚀 Quick Start

### 1. Installation

```bash
cd rl_prompt_engine
poetry install
```

### 2. Complete Example: Train & Generate

Here's a complete example showing how to train a model and generate prompts:

#### Step 1: Train a Model
```bash
# Train with default configuration (appointment booking)
poetry run python -m rl_prompt_engine.cli_simple train --timesteps 2000 --save-path models/demo_model
```

**Output:**
```
🚀 Training RL model...
Eval num_timesteps=1000, episode_reward=0.64 +/- 0.24
Episode length: 8.40 +/- 3.20
✅ Training completed. Model saved as models/demo_model
```

#### Step 2: List Available Options
```bash
# See what context types, stages, and components are available
poetry run python -m rl_prompt_engine.cli_simple list
```

**Output:**
```
🎯 Context Types:
  0: new_customer
  1: returning_customer
  2: price_sensitive
  3: premium_customer
  4: urgent_customer
  5: skeptical_customer

📈 Conversation Stages:
  0: opening
  1: exploration
  2: presentation
  3: negotiation
  4: closing
  5: follow_up

⚡ Urgency Levels:
  0: low
  1: medium
  2: high
```

#### Step 3: Generate Prompts
```bash
# Generate a greeting for new customers
poetry run python -m rl_prompt_engine.cli_simple generate \
  --model-path models/demo_model \
  --context-type 0 \
  --conversation-stage 0 \
  --urgency-level 0 \
  --output new_customer_greeting.txt

# Generate a closing for premium customers
poetry run python -m rl_prompt_engine.cli_simple generate \
  --model-path models/demo_model \
  --context-type 3 \
  --conversation-stage 4 \
  --urgency-level 2 \
  --output premium_customer_closing.txt
```

**Generated Prompt Example:**
```
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
- budget: Customer's budget range
- timeline: When they need the solution
- benefit: Main benefit of the product/service
- rate: Special rate or offer
- incentive: Special offer or incentive
- concern: Specific concern to address
- time_reference: When to schedule (e.g., today, this week, when convenient)

INSTRUCTIONS:
1. Fill in all parameters with appropriate values
2. Use the Friendly, patient, reassuring tone throughout
3. Incorporate all template parts in a natural flow
4. Keep the message conversational and professional
5. End with a clear call-to-action
6. Make it sound like a real conversation

Generate a complete message that follows this template and incorporates all specified elements.
```

#### What You Get
The generated prompts are **meta-prompts** that you can use with any LLM (ChatGPT, Claude, etc.) to generate actual customer messages. The RL agent learns to combine the right prompt components for each specific context.

**Example Usage with ChatGPT:**
1. Copy the generated prompt
2. Paste it into ChatGPT with your specific parameters
3. Get a personalized message for your customer

### 3. Alternative: Use Different Configurations

```bash
# Train with luxury car sales configuration
poetry run python -m rl_prompt_engine.cli_simple train \
  --config configs/luxury_config.json \
  --timesteps 5000 \
  --save-path models/luxury_model

# Generate with luxury model
poetry run python -m rl_prompt_engine.cli_simple generate \
  --model-path models/luxury_model \
  --context-type 0 \
  --conversation-stage 0 \
  --urgency-level 0 \
  --output luxury_greeting.txt
```

## 🎯 Features

- **🤖 RL-Powered**: Uses PPO reinforcement learning to optimize prompt strategies
- **🎭 Context Types**: Supports any customer/user personas (new customers, premium users, etc.)
- **💬 Conversation Stages**: Adapts to any conversation flow (opening, exploration, closing, etc.)
- **⚡ Urgency Levels**: Handles different urgency scenarios (low, medium, high)
- **🔧 Fully Configurable**: Customize everything via JSON configuration files
- **📊 Comprehensive Logging**: Detailed training progress and performance metrics
- **🖥️ Simple CLI**: Easy-to-use command line interface
- **🐍 Python API**: Clean, intuitive Python interface

## 📁 Project Structure

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

## 🎮 Usage Examples

### Python API

```python
from rl_prompt_engine import PromptSystem

# Create system with custom config
system = PromptSystem(config_file='configs/my_custom_config.json')

# Train a model
system.train(total_timesteps=10000, save_path='models/my_model')

# Generate prompt template
template = system.generate_meta_prompt(
    context_type=0,      # new_customer
    conversation_stage=3, # closing
    urgency_level=2      # high
)

print(template)
```

### Command Line Interface

```bash
# List available context types, stages, and components
poetry run python -m rl_prompt_engine.cli_simple list

# Train a new model
poetry run python -m rl_prompt_engine.cli_simple train \
  --config configs/my_custom_config.json \
  --timesteps 10000 \
  --save-path models/my_custom_model

# Generate templates for different scenarios
poetry run python -m rl_prompt_engine.cli_simple generate \
  --context-type 2 \
  --conversation-stage 3 \
  --urgency-level 2 \
  --output premium_customer_closing.txt

# Generate configuration with AI
poetry run python -m rl_prompt_engine.cli_simple config \
  --description "E-commerce customer support system" \
  --output configs/ecommerce_config.json
```

## ⚙️ Configuration

### Generic Configuration Structure

The system is fully configurable via JSON files. Here's the structure:

```json
{
  "max_prompt_length": 6,
  "max_turns": 10,
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
  },
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
  },
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
  },
  "urgency_levels": {
    "low": {
      "description": "Low urgency scenario",
      "time_reference": "when convenient",
      "approach": "Patient, no pressure",
      "preferred_components": ["greeting", "needs_assessment", "value_proposition"]
    },
    "high": {
      "description": "High urgency scenario",
      "time_reference": "today or tomorrow",
      "approach": "Urgent, action-oriented",
      "preferred_components": ["urgency_creation", "call_to_action", "incentive_offering"]
    }
  }
}
```

### Generate Configuration with AI

```bash
# Generate config from natural language description
poetry run python -m rl_prompt_engine.cli_simple config \
  --description "SaaS customer onboarding system for B2B software" \
  --output configs/saas_onboarding.json
```

## 🧠 How It Works

1. **Environment Setup**: The RL environment simulates any conversation scenario with configurable context types, conversation stages, and urgency levels.

2. **RL Training**: The PPO algorithm learns to select optimal combinations of prompt components based on the context.

3. **Template Generation**: The trained model generates contextually appropriate prompt templates by selecting and combining components.

4. **Meta-Prompt Creation**: The system creates meta-prompts that can be used with LLMs to generate actual messages.

## 📊 Training Parameters

The system uses optimized PPO parameters:
- **Learning Rate**: 3e-4
- **Batch Size**: 512
- **Entropy Coefficient**: 0.1 (encourages exploration)
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2

## 🎯 Use Cases

The generic design makes it suitable for:

- **Sales**: Appointment booking, product sales, lead generation
- **Support**: Customer service, technical support, troubleshooting
- **Marketing**: Email campaigns, social media, content creation
- **Onboarding**: User onboarding, training, education
- **Retention**: Customer retention, upselling, cross-selling

## 🔧 Development

```bash
# Install dependencies
poetry install

# Format code
poetry run black .

# Lint code
poetry run flake8

# Run type checking
poetry run mypy rl_prompt_engine/
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Make sure you're running commands from the correct directory
2. **Model not found**: Train a model first using the `train` command
3. **Config errors**: Check your JSON configuration file for syntax errors
4. **OpenAI API errors**: Ensure your API key is correctly set in the `.env` file (only needed for AI config generation)

### Getting Help

- Check the logs in the `logs/` directory for detailed error information
- Use `--help` flag with any CLI command for usage information
- Review the configuration files in `configs/` for examples
- Use the `list` command to see available options

## 🚀 Migration from v1.0

If you're upgrading from the old appointment-specific version:

1. **Update imports**: Use `PromptSystem` instead of `MetaPromptingSystem`
2. **Update CLI**: Use `cli_simple.py` instead of `cli.py`
3. **Update configs**: Use the new generic configuration format
4. **Update API calls**: Use `generate_meta_prompt()` instead of `generate_prompt_template()`

The new system is **backward compatible** and much simpler to use!
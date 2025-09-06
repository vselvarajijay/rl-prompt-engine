# 🤖 RL Prompt Engine

A reinforcement learning-powered system for generating prompt templates for appointment booking conversations.

## 🚀 Quick Start

### 1. Installation

```bash
cd /Users/vijayselvaraj/Development/rl-sales-agent/rl_prompt_engine
poetry install
```

### 2. Environment Setup

```bash
# Run the setup script
poetry run python setup.py

# Or manually:
cp env.template .env
# Edit .env and add your OpenAI API key
```

### 3. Basic Usage

```python
from rl_prompt_engine import MetaPromptingSystem

# Create and train a system
system = MetaPromptingSystem()
system.train(total_timesteps=1000)

# Generate a prompt template
template = system.generate_prompt_template(
    customer_type=0,  # cautious
    conversation_stage=0,  # early
    urgency_level=0  # low
)
print(template)
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Detailed usage instructions
- **[Usage Examples](usage_examples.py)** - Comprehensive examples
- **[Environment Template](env.template)** - Environment variables reference

## 🎯 Features

- **RL-Powered**: Uses reinforcement learning to optimize prompt strategies
- **Customer Segmentation**: Supports 6 different customer types
- **Conversation Stages**: Adapts to early, middle, late, and closing stages
- **Urgency Levels**: Handles low, medium, and high urgency scenarios
- **OpenAI Integration**: Generates actual prompts using OpenAI's API
- **Command Line Interface**: Easy-to-use CLI for training and generation

## 🛠️ Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint code
poetry run flake8
```

## 📝 License

This project is licensed under the MIT License.

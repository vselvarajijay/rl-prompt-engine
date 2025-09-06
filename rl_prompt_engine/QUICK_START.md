# 🚀 Quick Start Guide

## Installation

```bash
cd /Users/vijayselvaraj/Development/rl-sales-agent/rl_prompt_engine
poetry install
```

## Environment Setup

1. **Copy the environment template:**
   ```bash
   cp env.template .env
   ```

2. **Edit the .env file with your OpenAI API key:**
   ```bash
   # Required for OpenAI integration
   OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **Optional: Configure training parameters:**
   ```bash
   # Training configuration (optional)
   N_ENVS=8
   TOTAL_STEPS=300000
   EVAL_FREQ=15000
   ```

## Basic Usage

### 1. Simple Template Generation

```python
from rl_prompt_engine import MetaPromptingSystem

# Create system
system = MetaPromptingSystem()

# Generate a prompt template
template = system.generate_prompt_template(
    customer_type=0,  # cautious
    conversation_stage=0,  # early
    urgency_level=0  # low
)

print(template)
```

### 2. Training Your Own Model

```python
from rl_prompt_engine import train_system

# Train a new model
system = train_system(total_timesteps=10000, save_path="my_model")

# Use the trained model
template = system.generate_prompt_template(
    customer_type=2,  # ready_buyer
    conversation_stage=2,  # late
    urgency_level=2  # high
)
```

### 3. Multiple Scenarios

```python
from rl_prompt_engine import MetaPromptingSystem

system = MetaPromptingSystem()

# Define scenarios
scenarios = [
    {"customer_type": 0, "conversation_stage": 0, "urgency_level": 0},
    {"customer_type": 1, "conversation_stage": 1, "urgency_level": 1},
    {"customer_type": 2, "conversation_stage": 2, "urgency_level": 2}
]

# Generate templates for all scenarios
templates = system.generate_templates_for_scenarios(scenarios)

for template_data in templates:
    print(f"Template: {template_data['template']}")
```

## Command Line Interface

### Train a model
```bash
poetry run python -m rl_prompt_engine.cli train --timesteps 5000
```

### Generate a template
```bash
# First train a model (required)
poetry run python -m rl_prompt_engine.cli train --timesteps 1000

# Then generate templates
poetry run python -m rl_prompt_engine.cli generate --customer-type 0 --stage 0 --urgency 0
```

### Generate multiple templates
```bash
poetry run python -m rl_prompt_engine.cli batch
```

### Get help
```bash
poetry run python -m rl_prompt_engine.cli help
```

## Customer Types

- `0` - Cautious
- `1` - Price Shopper  
- `2` - Ready Buyer
- `3` - Research Buyer
- `4` - Impulse Buyer
- `5` - Skeptical

## Conversation Stages

- `0` - Early
- `1` - Middle
- `2` - Late
- `3` - Closing

## Urgency Levels

- `0` - Low
- `1` - Medium
- `2` - High

## Examples

Run the examples to see the package in action:

```bash
poetry run python usage_examples.py
```

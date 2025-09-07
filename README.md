# RL Prompt Engine

A clean, simple RL system for learning prompt construction strategies using PPO (Proximal Policy Optimization) from Stable Baselines3.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rl-prompt-engine

# Install dependencies
poetry install

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Make sure you're in the project root directory
cd /path/to/rl-prompt-engine

# Train a PPO model
poetry run python -m rl_prompt_engine.cli train --timesteps 10000

# Generate a prompt
poetry run python -m rl_prompt_engine.cli generate --context-type 0 --conversation-stage 0 --urgency-level 0 --custom-vars '{"first_name": "John", "product": "our service", "company_name": "Acme Corp", "budget": "your budget", "timeline": "next month", "benefit": "efficiency", "rate": "20% off", "incentive": "free trial", "concern": "implementation", "time_reference": "this week"}'

# List available options
poetry run python -m rl_prompt_engine.cli list

# Template management
poetry run python -m rl_prompt_engine.cli template list
poetry run python -m rl_prompt_engine.cli template show
poetry run python -m rl_prompt_engine.cli template validate
```

## Python API

```python
from rl_prompt_engine import PromptEngine, PromptEnv

# Create and train a model
engine = PromptEngine("rl_prompt_engine/configs/generic_config.json")
model = engine.train(total_timesteps=10000)

# Generate a strategy
strategy = engine.generate_strategy(
    context_type=0,      # new_customer
    conversation_stage=0, # opening
    urgency_level=0      # low
)
print(f"Selected components: {strategy}")

# Generate a full template
template = engine.generate_template(
    context_type=0,
    conversation_stage=0,
    urgency_level=0,
    custom_variables={
        "first_name": "John",
        "product": "our premium service"
    }
)
print(template)
```

## What It Does

The PPO agent learns to select the best prompt components based on:
- **Context Type** (new_customer, premium_customer, price_sensitive, etc.)
- **Conversation Stage** (opening, exploration, presentation, closing, etc.)  
- **Urgency Level** (low, medium, high)

The agent receives rewards for:
- Selecting effective components for the given context
- Matching components to conversation stage
- Creating urgency-appropriate prompts
- Building efficient, concise prompts

## Key Features

- ✅ **Stable Baselines3 PPO** - Industry-standard RL implementation
- ✅ **Configurable Environment** - JSON-based configuration system
- ✅ **Context-Aware Learning** - Adapts to different customer types and situations
- ✅ **Clean API** - Simple Python interface for training and generation
- ✅ **CLI Interface** - Easy command-line training and testing
- ✅ **Focused Design** - Does one thing well: learn optimal prompt construction

## Technical Details

- **Algorithm**: PPO (Proximal Policy Optimization) from Stable Baselines3
- **Environment**: Custom Gymnasium environment with discrete actions
- **Observation Space**: Flattened array with context features and component history
- **Action Space**: Discrete selection of prompt components + finish action
- **Reward**: Context-aware effectiveness scoring with efficiency bonuses

## 📁 Project Structure

```
rl-prompt-engine/
├── rl_prompt_engine/           # Main package
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # Command line interface
│   ├── core/                  # Core functionality
│   │   ├── __init__.py        # Core package init
│   │   ├── prompt_engine.py   # Main RL system with PPO
│   │   ├── prompt_env.py      # RL environment
│   │   ├── logging_config.py  # Logging setup
│   │   └── template_loader.py # Template management
│   ├── configs/               # Configuration files
│   │   └── generic_config.json # Generic configuration
│   ├── templates/             # Markdown templates
│   │   ├── meta_prompt_template.md # Meta prompt template (required)
│   │   └── prompt_evolution_template.md # Evolution examples (optional)
│   ├── models/                # Trained RL models
│   └── logs/                  # Training logs
├── pyproject.toml             # Poetry configuration
├── poetry.lock               # Dependency lock file
└── README.md                 # This file
```

## Configuration

**⚠️ Important**: All configuration values are required. The system will fail with explicit error messages if any configuration is missing.

The system uses `rl_prompt_engine/configs/generic_config.json` for configuration. You can customize:

- **Prompt Components**: Available building blocks (greeting, needs_assessment, etc.)
- **Context Types**: Customer categories (new_customer, premium_customer, etc.)
- **Conversation Stages**: Interaction phases (opening, exploration, closing, etc.)
- **Urgency Levels**: Time sensitivity (low, medium, high)
- **Component Effectiveness**: Base effectiveness scores for each component
- **Context Preferences**: Which components work best for each context

### Available Options

Run `poetry run python -m rl_prompt_engine.cli list` to see all available:

**Context Types:**
- `0`: new_customer
- `1`: returning_customer  
- `2`: price_sensitive
- `3`: premium_customer
- `4`: urgent_customer
- `5`: skeptical_customer

**Conversation Stages:**
- `0`: opening
- `1`: exploration
- `2`: presentation
- `3`: negotiation
- `4`: closing
- `5`: follow_up

**Urgency Levels:**
- `0`: low
- `1`: medium
- `2`: high

**Prompt Components:**
- `0`: greeting
- `1`: needs_assessment
- `2`: value_proposition
- `3`: objection_handling
- `4`: urgency_creation
- `5`: social_proof
- `6`: incentive_offering
- `7`: call_to_action
- `8`: reassurance
- `9`: follow_up

## Template Customization

The meta prompt template is now configurable via markdown files! You can customize the prompt structure without touching the core engine code.

### Template Location
- **Default template**: `rl_prompt_engine/templates/meta_prompt_template.md`
- **Template variables**: All `{variable_name}` placeholders are automatically filled

### Available Template Variables
- `{context_type_name}` - Customer type (e.g., "new_customer")
- `{stage_name}` - Conversation stage (e.g., "opening")
- `{urgency_name}` - Urgency level (e.g., "low")
- `{context_description}` - Customer description from config
- `{context_tone}` - Recommended tone from config
- `{context_approach}` - Recommended approach from config
- `{urgency_time_reference}` - Time reference from config
- `{full_template}` - Generated component template

### Template Management Commands
```bash
# List available templates
poetry run python -m rl_prompt_engine.cli template list

# Show template content
poetry run python -m rl_prompt_engine.cli template show

# Validate template variables
poetry run python -m rl_prompt_engine.cli template validate

# Show specific template
poetry run python -m rl_prompt_engine.cli template show --template meta_prompt_template
```

### Creating Custom Templates
1. Create a new `.md` file in `rl_prompt_engine/templates/`
2. Use the template variables listed above
3. The system will automatically load and use your custom template

### Required Templates
The system requires these templates to function:
- **`meta_prompt_template.md`** - Main prompt generation template (required)
- **`prompt_evolution_template.md`** - Prompt evolution examples (optional)

If templates are missing, the system will fail gracefully with clear error messages.

## Troubleshooting

### "Config file not found" Error
If you get a `FileNotFoundError: Config file not found`, make sure you're running commands from the project root directory:

```bash
# Check you're in the right directory
pwd
# Should show: /path/to/rl-prompt-engine

# If you're in rl_prompt_engine/ subdirectory, go back up
cd ..

# Then run commands
poetry run python -m rl_prompt_engine.cli train --timesteps 1000
```

### Missing Custom Variables
The generate command requires custom variables. If you get a "custom_variables must be provided" error, make sure to include the `--custom-vars` parameter:

```bash
poetry run python -m rl_prompt_engine.cli generate --custom-vars '{"first_name": "John", "product": "service", ...}'
```

## Example Output

When you run the generate command, you'll see output like:

```
✅ Generated Prompt Template:
============================================================
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
Hello {first_name}, how can we assist you today?

I'd like to understand your specific needs for {product}

PARAMETERS TO FILL:
- first_name: Customer's first name
- product: Specific product/service they're interested in
- company_name: Name of your company
...

INSTRUCTIONS:
1. Fill in all parameters with appropriate values
2. Use the Friendly, patient, reassuring tone throughout
3. Incorporate all template parts in a natural flow
4. Keep the message conversational and professional
5. End with a clear call-to-action
6. Make it sound like a real conversation

Generate a complete message that follows this template and incorporates all specified elements.
============================================================
```

## Training Progress

During training, you'll see:
- Real-time training progress
- Episode rewards and lengths
- Training logs with detailed metrics

## Requirements

- Python 3.9+
- Poetry (recommended) or pip
- Stable Baselines3
- Gymnasium
- NumPy, Matplotlib, Seaborn

That's it! Simple, clean, and focused on what you actually need.
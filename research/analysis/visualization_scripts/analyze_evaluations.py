#!/usr/bin/env python3
"""
Script to analyze evaluation data from trained RL models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_evaluation_data(data_dir):
    """Load evaluation data from .npz files."""
    eval_files = list(data_dir.glob("evaluation_results/*.npz"))
    
    data = {}
    for file_path in eval_files:
        model_name = file_path.stem
        data[model_name] = np.load(file_path)
        print(f"Loaded {model_name}:")
        for key in data[model_name].keys():
            print(f"  {key}: {data[model_name][key].shape}")
    
    return data

def plot_learning_curves(data, output_dir):
    """Plot learning curves for different models."""
    plt.figure(figsize=(12, 8))
    
    for model_name, model_data in data.items():
        if 'timesteps' in model_data and 'results' in model_data:
            timesteps = model_data['timesteps']
            results = model_data['results']
            
            # Plot mean reward over time
            plt.subplot(2, 2, 1)
            plt.plot(timesteps, results.mean(axis=1), label=model_name)
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.title('Learning Curves')
            plt.legend()
            
            # Plot success rate over time
            plt.subplot(2, 2, 2)
            success_rate = (results > 0).mean(axis=1)
            plt.plot(timesteps, success_rate, label=model_name)
            plt.xlabel('Timesteps')
            plt.ylabel('Success Rate')
            plt.title('Success Rate Over Time')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_component_usage(config_file, output_dir):
    """Analyze prompt component usage patterns."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    components = list(config['prompt_components'].keys())
    effectiveness_scores = [config['prompt_components'][comp]['effectiveness'] 
                           for comp in components]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(components)), effectiveness_scores)
    plt.xlabel('Prompt Components')
    plt.ylabel('Effectiveness Score')
    plt.title('Component Effectiveness Scores')
    plt.xticks(range(len(components)), components, rotation=45, ha='right')
    
    # Color bars by effectiveness
    colors = plt.cm.viridis(np.array(effectiveness_scores))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_system_architecture_diagram(output_dir):
    """Create a simple system architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define components
    components = {
        'Environment': (2, 6),
        'PPO Agent': (6, 6),
        'Generator': (10, 6),
        'Context': (2, 3),
        'Actions': (6, 3),
        'Templates': (10, 3)
    }
    
    # Draw boxes
    for name, (x, y) in components.items():
        rect = plt.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows
    arrows = [
        ((2, 5.6), (5.2, 5.6)),  # Environment -> PPO
        ((6.8, 5.6), (9.2, 5.6)),  # PPO -> Generator
        ((2, 2.6), (5.2, 2.6)),  # Context -> Actions
        ((6.8, 2.6), (9.2, 2.6)),  # Actions -> Templates
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RL Prompt Generation System Architecture', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / 'paper' / 'data'
    output_dir = script_dir.parent.parent / 'paper' / 'figures' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting evaluation data analysis...")
    
    # Load data
    eval_data = load_evaluation_data(data_dir)
    
    if eval_data:
        # Plot learning curves
        plot_learning_curves(eval_data, output_dir)
        
        # Analyze component usage
        config_file = data_dir / 'luxury_config.json'
        if config_file.exists():
            analyze_component_usage(config_file, output_dir)
        
        # Create architecture diagram
        create_system_architecture_diagram(output_dir)
        
        print(f"Analysis complete! Figures saved to {output_dir}")
    else:
        print("No evaluation data found. Please train models first.")

if __name__ == "__main__":
    main()

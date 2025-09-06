import argparse
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from purchase_env import AutomotiveAppointmentEnv

def evaluate(model, episodes=200):
    """
    Evaluate a trained AI model by having it play multiple games and measuring its performance.
    
    In reinforcement learning, we train an AI agent to make decisions by having it interact
    with an environment (like a game). This function tests how well the trained agent performs.
    
    Args:
        model: A pre-trained AI model that knows how to make decisions
        episodes: Number of complete games to play for testing (default: 200)
    
    Returns:
        win_rate: Percentage of games the AI won (0.0 to 1.0)
        action_counts: How often the AI chose each possible action
    """
    # Create the environment (the "game" the AI will play)
    # n_personas=6 means there are 6 different types of customers the AI might encounter
    env = AutomotiveAppointmentEnv(n_personas=6)
    
    # Keep track of how many games the AI won
    successes = 0
    
    # Keep track of which actions the AI chose most often
    # This helps us understand the AI's strategy
    action_counts = Counter()
    
    # Play multiple games to get a good measure of performance
    for _ in range(episodes):
        # Start a new game - reset the environment to initial state
        # obs = "observation" - what the AI can see about the current situation
        obs, _ = env.reset()
        done = False  # Game continues until this becomes True
        reward = 0.0  # Points the AI earns (1.0 = win, 0.0 = lose)
        
        # Keep playing until the game ends
        while not done:
            # Ask the AI what action it wants to take based on what it can see
            # deterministic=True means the AI always picks its best choice (no randomness)
            action, _ = model.predict(obs, deterministic=True)
            
            # Record which action the AI chose
            action_counts[int(action)] += 1
            
            # Execute the action and see what happens
            # The environment tells us: new observation, reward, if game ended, and extra info
            obs, reward, done, truncated, info = env.step(int(action))
        
        # If the final reward is 1.0, the AI won this game
        successes += int(reward == 1.0)
    
    # Calculate win rate as percentage of games won
    return successes / episodes, action_counts

if __name__ == "__main__":
    # Set up command line arguments so users can customize the evaluation
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to evaluate")
    parser.add_argument("--model", type=str, default="ppo_purchase_sparse", help="Model path")
    args = parser.parse_args()
    
    # Load the pre-trained AI model from a file
    # PPO (Proximal Policy Optimization) is the algorithm used to train this AI
    model = PPO.load(args.model)
    
    # Run the evaluation and get results
    win_rate, action_counts = evaluate(model, episodes=args.episodes)
    
    # Display the results in a human-readable format
    print(f"Win rate over {args.episodes} episodes: {win_rate:.3f}")
    print("\nAction distribution:")
    
    # Calculate total actions taken across all games
    total_actions = sum(action_counts.values())
    
    # Show how often the AI chose each possible action
    # This helps us understand the AI's strategy and decision-making patterns
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        percentage = (count / total_actions) * 100
        print(f"  Action {action}: {count:4d} ({percentage:5.1f}%)")

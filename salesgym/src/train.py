# Reinforcement Learning Training Script for Sales Agent
# This script trains a PPO (Proximal Policy Optimization) agent to learn sales strategies
# in a simulated purchase environment with multiple customer personas.

import os
# os module allows us to read environment variables for configuration
from stable_baselines3 import PPO
# PPO is a state-of-the-art policy gradient method that's stable and sample-efficient
from stable_baselines3.common.env_util import make_vec_env
# make_vec_env creates multiple parallel environments for faster training
from purchase_env import AutomotiveAppointmentEnv
# Our custom environment that simulates sales interactions with different customer types

def main():
    # Configuration: Read from environment variables with sensible defaults
    # This allows us to easily experiment with different settings without changing code
    n_envs = int(os.environ.get("N_ENVS", 16))  # Number of parallel environments
    total_steps = int(os.environ.get("TOTAL_STEPS", 500_000))  # Total training steps

    # Create vectorized environment: Multiple parallel instances of our sales environment
    # Each environment runs independently, allowing us to collect more experience per second
    # The lambda function ensures each environment gets a fresh instance with 6 customer personas
    env = make_vec_env(lambda: AutomotiveAppointmentEnv(n_personas=6), n_envs=n_envs)
    
    # Initialize PPO model with carefully tuned hyperparameters
    # PPO is particularly good for continuous control and complex environments
    model = PPO(
        "MultiInputPolicy",  # Policy type that can handle different observation spaces
        env,                 # The environment to train on
        learning_rate=3e-4,  # How fast the model learns (0.0003)
        n_steps=1024,        # Steps to collect before updating policy
        batch_size=256,      # Size of training batches
        gamma=0.995,         # Discount factor: how much we value future rewards
        gae_lambda=0.95,     # GAE parameter for advantage estimation
        clip_range=0.2,      # PPO clipping parameter for stable updates
        ent_coef=0.01,       # Entropy bonus to encourage exploration
        verbose=1,           # Show training progress
    )
    
    # Start training: The agent will interact with the environment and learn
    # This is where the magic happens - the agent tries different actions,
    # receives rewards, and gradually improves its strategy
    model.learn(total_timesteps=total_steps)
    
    # Save the trained model for later use (inference or further training)
    model.save("ppo_purchase_sparse")
    print("Saved model to ppo_purchase_sparse.zip")

# Standard Python idiom: only run main() when script is executed directly
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Custom Training Callback for Enhanced Logging
"""

from stable_baselines3.common.callbacks import BaseCallback
from .logging_config import get_logger
import numpy as np


class DetailedTrainingCallback(BaseCallback):
    """
    Custom callback that provides detailed logging during training.
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.training_logger = get_logger('training')
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to env.step().
        """
        # Log training progress with simple progress indicator
        if self.n_calls % self.log_freq == 0:
            progress = (self.n_calls / 5000) * 100  # Assuming 5000 total timesteps
            print(f"🔄 Training progress: {self.n_calls}/5000 steps ({progress:.1f}%)")
            self.training_logger.info(f"Training step {self.n_calls} - Progress: {progress:.1f}%")
            
            # Log current episode info if available
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get current episode info from the environment
                    episode_rewards = self.training_env.get_attr('episode_rewards')
                    episode_lengths = self.training_env.get_attr('episode_lengths')
                    
                    if episode_rewards and episode_rewards[0]:
                        avg_reward = np.mean(episode_rewards[0][-100:])  # Last 100 episodes
                        avg_length = np.mean(episode_lengths[0][-100:]) if episode_lengths[0] else 0
                        
                        print(f"📊 Avg reward: {avg_reward:.3f}, Avg length: {avg_length:.1f}")
                        self.training_logger.info(f"Average reward (last 100): {avg_reward:.3f}")
                        self.training_logger.info(f"Average episode length (last 100): {avg_length:.1f}")
                except Exception as e:
                    self.training_logger.debug(f"Could not get episode info: {e}")
        
        return True
    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interactions using the current model.
        """
        self.training_logger.debug("Starting new rollout")
    
    def _on_rollout_end(self) -> None:
        """
        This method is called when the rollout is completed.
        """
        self.training_logger.debug("Rollout completed")
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_logger.info("Training started")
    
    def _on_training_end(self) -> None:
        """
        This method is called before training ends.
        """
        self.training_logger.info("Training ended")

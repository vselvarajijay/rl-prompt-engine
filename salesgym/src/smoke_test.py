#!/usr/bin/env python3
"""
Smoke test script to verify the environment and training work correctly.
"""
import numpy as np
from purchase_env import AutomotiveAppointmentEnv, N_ACTIONS

def test_env_spaces():
    """Test that environment spaces are correctly defined."""
    print("Testing environment spaces...")
    env = AutomotiveAppointmentEnv(n_personas=3, seed=42)
    
    # Test observation space
    obs, _ = env.reset()
    assert "features" in obs, "Missing 'features' in observation"
    assert obs["features"].shape == (5,), f"Expected features shape (5,), got {obs['features'].shape}"
    assert 0 <= obs["persona_id"] < 3, f"Invalid persona_id: {obs['persona_id']}"
    assert obs["last_action"] == N_ACTIONS, f"Expected last_action={N_ACTIONS}, got {obs['last_action']}"
    
    print("✓ Observation space test passed")

def test_step_reset_shape():
    """Test that step and reset return correct shapes."""
    print("Testing step/reset shapes...")
    env = AutomotiveAppointmentEnv(n_personas=3, seed=1)
    
    obs, info = env.reset()
    assert isinstance(obs, dict), "Reset should return dict observation"
    assert isinstance(info, dict), "Reset should return dict info"
    
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict), "Step should return dict observation"
    assert isinstance(reward, (int, float)), "Step should return numeric reward"
    assert isinstance(terminated, bool), "Step should return bool terminated"
    assert isinstance(truncated, bool), "Step should return bool truncated"
    assert isinstance(info, dict), "Step should return dict info"
    
    print("✓ Step/reset shape test passed")

def test_training_runs():
    """Test that training runs for a few steps without errors."""
    print("Testing training runs...")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create a small training setup
        env = make_vec_env(lambda: AutomotiveAppointmentEnv(n_personas=3), n_envs=2)
        model = PPO("MultiInputPolicy", env, verbose=0)
        
        # Train for just a few steps
        model.learn(total_timesteps=100)
        
        print("✓ Training test passed")
        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("Running SalesGym smoke tests...\n")
    
    try:
        test_env_spaces()
        test_step_reset_shape()
        training_ok = test_training_runs()
        
        print(f"\n{'='*50}")
        if training_ok:
            print("🎉 All smoke tests passed! The environment is ready to use.")
        else:
            print("⚠️  Environment tests passed, but training had issues.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

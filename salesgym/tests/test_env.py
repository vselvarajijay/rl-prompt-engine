import numpy as np
from src.purchase_env import PurchaseEnv, N_ACTIONS

def test_spaces_and_reset():
    env = PurchaseEnv(n_personas=3, seed=42)
    obs, _ = env.reset()
    assert "features" in obs and obs["features"].shape == (4,)
    assert 0 <= obs["persona_id"] < 3
    assert obs["last_action"] == N_ACTIONS

def test_step_runs_and_terminates():
    env = PurchaseEnv(n_personas=3, seed=1)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        action = np.random.randint(0, N_ACTIONS)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    assert done
    assert steps <= 100

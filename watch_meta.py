# watch_meta.py
# (FIXED: Added VecNormalize loading and fixed reset unpacking)

import time
import os
import argparse
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv 
import numpy as np

from src.env.meta_env import MetaEnv, KEY_EVENT_NAMES
from src.env.wrappers import DebugDashboardWrapper 
from src.utils.location_memory import LocationMemory 

# --- Constants ---
MODEL_ROOT = "models"
META_LESSON = "meta"
META_MODEL_NAME = "meta_model"
META_MODEL_PATH = os.path.join(MODEL_ROOT, META_LESSON, f"{META_MODEL_NAME}.zip")
VEC_NORMALIZE_PATH = os.path.join(MODEL_ROOT, META_LESSON, "vec_normalize.pkl") 

# --- Model Validation ---
if not os.path.exists(META_MODEL_PATH):
    print(f"Error: Meta-Manager Model not found at {META_MODEL_PATH}")
    print(f"Please run train_meta.py to train it first.")
    exit()
if not os.path.exists(VEC_NORMALIZE_PATH):
    print(f"Error: VecNormalize stats not found at {VEC_NORMALIZE_PATH}")
    print(f"The model was trained with normalization and cannot be loaded without this file.")
    exit()


# --- Environment Initialization ---
print("Creating visible Meta-Manager environment...")
base_env = MetaEnv(headless=False, location_memory=LocationMemory())
base_env = DebugDashboardWrapper(base_env) 

# Wrap the environment in a list for VecNormalize compatibility
env_list = [lambda: base_env] 
vec_env = DummyVecEnv(env_list) 

print("Environment created. Loading VecNormalize stats...")

# --- Load the VecNormalize Wrapper ---
try:
    env = VecNormalize.load(VEC_NORMALIZE_PATH, vec_env)
    env.training = False 
    env.norm_reward = False 
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load VecNormalize stats from {VEC_NORMALIZE_PATH}")
    print(f"Details: {e}")
    exit()


print("Loading Meta-Manager model...")
model = RecurrentPPO.load(META_MODEL_PATH, env=env, device="cpu")

print(f"--- META-MANAGER AI is ONLINE ---")
print(f"--- STARTING 'CEO' AI VISUAL TEST ---")

# --- Simulation Setup ---
# (FIXED) Use star-unpacking for reset
obs_vec, info_vec, *_ = env.reset()
obs = {k: v[0] for k, v in obs_vec.items()}
info = info_vec[0]
lstm_states = None 
total_reward = 0

try:
    while True:
        # --- 1. Meta-Manager (CEO) Prediction ---
        action, lstm_states = model.predict(
            obs_vec, 
            state=lstm_states, 
            deterministic=True
        )
        action_int = action.item()
        chosen_goal = KEY_EVENT_NAMES[action_int]
        
        print(f"\n==============================================")
        print(f"   CEO AI Goal: [{chosen_goal.upper()}]")
        print(f"==============================================\n")
        
        # --- 2. Meta-Manager Step Execution ---
        start_time = time.time()
        
        # Step the wrapped environment
        obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = env.step(np.array([action_int]))
        
        # Unpack for local display/logic
        obs = {k: v[0] for k, v in obs_vec.items()}
        reward = reward_vec[0]
        terminated = terminated_vec[0]
        truncated = truncated_vec[0]
        info = info_vec[0]
        
        end_time = time.time()
        total_reward += reward
        
        # --- 3. Logging ---
        print(f"\n--- Goal Result ---")
        if reward > 0:
            print(f"  ✅ GOAL ACHIEVED: [{chosen_goal.upper()}]")
        else:
            print(f"  ❌ GOAL FAILED OR TIMED OUT")
            
        print(f"  Reward: {reward:+.2f}")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Total Reward: {total_reward:.2f}")
        
        # --- 4. Episode Reset ---
        if terminated or truncated:
            print("\nEpisode finished. Resetting...")
            obs_vec, info_vec, *_ = env.reset() # (FIXED) Apply star-unpacking here
            obs = {k: v[0] for k, v in obs_vec.items()}
            info = info_vec[0]
            # We DON'T reset LSTM states here
            
except KeyboardInterrupt:
    print("\nWatcher interrupted by user. Exiting.")
finally:
    env.close()
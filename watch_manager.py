# watch_manager.py
# (FINAL FIX: Added VecNormalize loading and fixed reset unpacking)

import time
from sb3_contrib import RecurrentPPO
import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv 
from stable_baselines3.common.utils import set_random_seed 
import argparse # Needed for this script to run standalone

from src.env.manager_env import ManagerEnv 
from src.env.wrappers import DebugDashboardWrapper # Retained for non-VecEnv compatibility
from src.utils.location_memory import LocationMemory 

# --- Constants and Setup ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")
MODEL_ROOT = "models"

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Watch a trained Manager AI model.")
parser.add_argument(
    "--model_path", 
    type=str, 
    default=None, 
    help="Path to the model (.zip) to load. If not specified, defaults to the latest manager model."
)
args = parser.parse_args()

# Determine model and VecNormalize paths
if args.model_path:
    MANAGER_MODEL_PATH = args.model_path
    # Derive VEC_NORMALIZE_PATH from model_path
    model_dir = os.path.dirname(MANAGER_MODEL_PATH)
    VEC_NORMALIZE_PATH = os.path.join(model_dir, "exploration_model_vec_normalize.pkl")
    print(f"Using specified model: {MANAGER_MODEL_PATH}")
    print(f"Derived VecNormalize path: {VEC_NORMALIZE_PATH}")
else:
    MANAGER_LESSON = "manager"
    MANAGER_MODEL_NAME = "manager_model"
    MANAGER_MODEL_PATH = os.path.join(MODEL_ROOT, MANAGER_LESSON, f"{MANAGER_MODEL_NAME}.zip")
    VEC_NORMALIZE_PATH = os.path.join(MODEL_ROOT, MANAGER_LESSON, "vec_normalize.pkl")
    print(f"Using default manager model: {MANAGER_MODEL_PATH}")
    print(f"Using default VecNormalize path: {VEC_NORMALIZE_PATH}")

# --- Utility function for creating the environment ---
def make_watch_env_fn(rank, seed=0, headless=False):
    def _init():
        env = ManagerEnv(
            rom_path=ROM_PATH, 
            state_path=STATE_PATH,
            headless=headless,
            location_memory=LocationMemory() # Always init memory
        )
        env = DebugDashboardWrapper(env) # Retained: ManagerEnv handles its own wrappers
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# --- Model Validation ---
if not os.path.exists(MANAGER_MODEL_PATH):
    print(f"Error: Manager Model not found at {MANAGER_MODEL_PATH}")
    print(f"Please ensure the path is correct or run train_manager.py to train it first.")
    exit()
if not os.path.exists(VEC_NORMALIZE_PATH):
    print(f"Error: VecNormalize stats not found at {VEC_NORMALIZE_PATH}")
    print(f"The model was trained with normalization and cannot be loaded without this file.")
    exit()

# --- Environment Initialization ---
print("Creating visible Manager environment...")

env_fns = [make_watch_env_fn(0, headless=False)]
vec_env = DummyVecEnv(env_fns) 

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

print(f"Loading Manager model from {MANAGER_MODEL_PATH}...")
model = RecurrentPPO.load(MANAGER_MODEL_PATH, env=env, device="cpu") 

print(f"--- MANAGER AI is ONLINE ---")
print(f"--- STARTING 'MANAGER AI' VISUAL TEST ---")

# --- Simulation Setup ---
def get_single_env_data(reset_output):
    """Utility to safely unpack VecEnv reset/step output."""
    try:
        obs_vec, info_vec, *_ = reset_output
    except ValueError:
        obs_vec, info_vec = reset_output
    
    if isinstance(obs_vec, dict):
        obs = {k: v[0] for k, v in obs_vec.items()}
    else:
        raise TypeError(f"Expected observation dict, got {type(obs_vec)}")

    info = info_vec[0]
    return obs_vec, info_vec, obs, info

# (FIXED) Use star-unpacking for initial reset
obs_vec, info_vec, obs, info = get_single_env_data(env.reset())
lstm_states = None 
specialist_names_list = env.unwrapped.SPECIALIST_NAMES

try:
    while True:
        # --- 1. Manager Prediction ---
        action, lstm_states = model.predict(
            obs_vec, 
            state=lstm_states, 
            deterministic=True
        )
        action_int = action.item()
        
        # Handle potential out-of-bounds
        if action_int < 0 or action_int >= len(specialist_names_list):
            print(f"Manager predicted invalid action {action_int}, defaulting to 'explore'")
            chosen_specialist = "explore"
            action_int = specialist_names_list.index("explore")
        else:
            chosen_specialist = specialist_names_list[action_int]
        
        # --- 2. Manager Step Execution ---
        start_time = time.time()
        print(f"\rManager chose: [{chosen_specialist.upper()}] - Running...", end="")
        
        # Step the wrapped environment
        step_output = env.step(np.array([action_int]))
        obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = step_output
        
        # Unpack for local display/logic
        obs = {k: v[0] for k, v in obs_vec.items()}
        reward = reward_vec[0]
        terminated = terminated_vec[0]
        truncated = truncated_vec[0]
        info = info_vec[0]
        
        end_time = time.time()
        
        # --- 3. Logging ---
        trigger = info.get("reward_trigger", "None")
        log_str = ""
        if trigger != "None":
            log_str = f"| Last Reward: {trigger}"
        
        print(f"\rManager action: [{chosen_specialist.upper():<10}] | "
              f"Accumulated Reward: {reward:+.2f} | "
              f"Time: {end_time - start_time:.2f}s {log_str}         ")
        
        # --- 4. Episode Reset ---
        if terminated or truncated:
            print("\nEpisode finished. Resetting...")
            obs_vec, info_vec, obs, info = get_single_env_data(env.reset())
            lstm_states = None 
            
except KeyboardInterrupt:
    print("\nWatcher interrupted by user. Exiting.")
finally:
    env.close()
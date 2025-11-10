# watch_specialist.py
# (FINAL FIX: Path resolution, unpacking robustness, AND delayed state loading)

import time
import os
import argparse 
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv 
from stable_baselines3.common.utils import set_random_seed 
import numpy as np
from pathlib import Path 

from src.env.pokemon_env import PokemonEnv
from src.env.wrappers import TransposeObservationWrapper, DebugDashboardWrapper
from src.utils.location_memory import LocationMemory 

# --- Constants and Setup ---
ROM_PATH = str(Path("PokemonYellow.gb").resolve())
STATE_DIR = str(Path("states").resolve())
STATE_PATH = str(Path(STATE_DIR) / "new_game.state")
MODEL_ROOT = str(Path("models").resolve())

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='exploration_model', 
                    help='Name of the model file to watch (e.g., exploration_model)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Full name of a specific checkpoint file to load (e.g., exploration_model_checkpoint_49994_steps.zip)')
args = parser.parse_args()

lesson_name = args.model_name.split('_')[0]

# (NEW) Determine the model path based on whether a checkpoint is specified
if args.checkpoint:
    MODEL_PATH = str(Path(MODEL_ROOT) / lesson_name / args.checkpoint)
else:
    MODEL_PATH = str(Path(MODEL_ROOT) / lesson_name / f"{args.model_name}.zip")
    
VEC_NORMALIZE_PATH = str(Path(MODEL_ROOT) / lesson_name / f"{args.model_name}_vec_normalize.pkl") 

# --- Model Validation ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print(f"Please run train_specialist.py --lesson {lesson_name} --model_name {args.model_name}")
    exit()
if not os.path.exists(VEC_NORMALIZE_PATH):
    print(f"Error: VecNormalize stats not found at {VEC_NORMALIZE_PATH}")
    print(f"The model was trained with normalization and cannot be loaded without this file.")
    exit()
    
# --- Utility function for creating the environment ---
def make_watch_env_fn(rank, seed=0, headless=False, lesson="exploration", abs_rom_path=ROM_PATH, abs_state_path=STATE_PATH):
    def _init():
        env = PokemonEnv(
            rom_path=abs_rom_path,
            state_path=abs_state_path, 
            headless=headless,
            lesson=lesson,
            location_memory=LocationMemory()
        )
        env = TransposeObservationWrapper(env)
        env = DebugDashboardWrapper(env) # Re-enabled for visual debugging
        
        # --- CRITICAL FIX: REMOVE THE INITIAL RESET CALL HERE ---
        # The VecNormalize.reset() call further down will handle it.
        # env.reset(seed=seed + rank) 
        
        return env
        
    set_random_seed(seed)
    return _init

# --- Environment Initialization ---
print(f"Creating visible environment (using '{STATE_PATH}')...")

env_fns = [make_watch_env_fn(0, headless=False, lesson=lesson_name)] 
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


# --- Model Loading ---
print(f"Loading model from {MODEL_PATH}...")
model = RecurrentPPO.load(MODEL_PATH, env=env, device="cpu") 
print(f"Recurrent (LSTM) model loaded: {args.model_name}.zip")
print(f"--- STARTING AI VISUAL TEST ---")

# --- Main Simulation Loop Setup ---
def get_single_env_data(reset_output):
    """Utility to safely unpack VecEnv reset/step output."""
    # Handle the case where VecNormalize.reset() returns only the observation dictionary
    if isinstance(reset_output, dict):
        obs_vec = reset_output
        info_vec = [{}]  # Create a dummy info list for compatibility
    else:
        # Handle tuple unpacking for other cases (like a raw env.reset())
        try:
            obs_vec, info_vec, *_ = reset_output
        except (ValueError, TypeError):
            obs_vec, info_vec = reset_output

    # At this point, obs_vec should be the observation dictionary
    if not isinstance(obs_vec, dict):
        raise TypeError(f"Expected observation dict, got {type(obs_vec)}")

    # Unpack the single environment's data from the batched VecEnv output
    obs = {k: v[0] for k, v in obs_vec.items()}
    info = info_vec[0] if info_vec else {}

    return obs_vec, info_vec, obs, info

try:
    # --- Initial Reset ---
    print(f"--- Calling VecNormalize.reset(), expecting state loading to succeed ---")
    obs_vec, info_vec, obs, info = get_single_env_data(env.reset())
except Exception as e:
    print(f"CRITICAL ERROR during initial VecNormalize.reset() and unpacking: {e}")
    raise

lstm_states = None
step_count = 0
try:
    while True:
        step_count += 1
        # --- Predict Action ---
        action, lstm_states = model.predict(
            obs_vec, 
            state=lstm_states, 
            deterministic=True
        )
        
        # --- Step Environment ---
        action_int = action.item()
        print(f"Step: {step_count}, Action: {action_int}")
        
        step_output = env.step(np.array([action_int])) 
        
        # Handle environments returning 4-tuple (obs, rew, done, info) or 5-tuple (obs, rew, terminated, truncated, info)
        if len(step_output) == 4:
            obs_vec, reward_vec, terminated_vec, info_vec = step_output
            truncated_vec = np.array([False] * len(terminated_vec))
        else:
            obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = step_output
        
        # Unpack batched results
        obs = {k: v[0] for k, v in obs_vec.items()}
        info = info_vec[0] 
        
        # Check termination flags
        terminated = terminated_vec[0]
        truncated = truncated_vec[0]
        
        time.sleep(0.01)
            
        # --- Episode Reset ---
        if terminated or truncated:
            print("\nEpisode finished. Resetting...")
            try:
                obs_vec, info_vec, obs, info = get_single_env_data(env.reset())
            except Exception as e:
                print(f"CRITICAL ERROR during episode reset/unpacking: {e}")
                raise
            
            lstm_states = None
            
except KeyboardInterrupt:
    print("\nWatcher interrupted by user. Exiting.")
finally:
    env.close()
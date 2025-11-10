# train_meta.py
# (FINAL VERSION: Stabilized RecurrentPPO training hyperparams for L3)

import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter 
import argparse
import json 

from src.env.meta_env import MetaEnv 
from src.env.wrappers import DebugDashboardWrapper
from src.utils.callbacks import LoggingCallback
from src.agents.policy import CustomCombinedExtractor
from train_specialist import ProgressBarCallback, RecurrentPPOWithProgress
from src.utils.location_memory import LocationMemory 

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")
LOG_ROOT = "logs"
MODEL_ROOT = "models"

# --- Hyperparameters (FIXED: n_epochs increased, n_steps increased) ---
seed = 0
n_epochs = 20    # Increased from 10 to 20 for stability
ent_coef = 0.01
gamma = 0.997 

NORM_OBS_KEYS = ["vision", "stats", "map", "menu_state", "items"]

# (NEW) Updated make_env_fn to accept location_memory
def make_env_fn(rank, seed=0, headless=True, location_memory=None):
    """
    Utility function for creating a parallel Meta-Manager environment instance.
    """
    def _init():
        env = MetaEnv(
            rom_path=ROM_PATH,
            state_path=STATE_PATH, 
            headless=headless,
            location_memory=location_memory # (NEW) Pass memory to env
        )
        
        if not headless:
            env = DebugDashboardWrapper(env)
            
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    """Main function to run the Meta-Manager training script."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', 
                        help='Run in slow, visible debug mode (1 CPU, headless=False)')
    parser.add_argument('--num_cpu', type=int, default=None, 
                        help='Number of parallel environments (default: os.cpu_count() - 2)')
    parser.add_argument('--total_timesteps', type=int, default=100_000, 
                        help='Total timesteps to train for')
    parser.add_argument('--n_steps', type=int, default=256, # Increased from 64 to 256
                        help='Steps per env per update (rollout buffer size)')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='PPO mini-batch size')
    args = parser.parse_args()

    lesson_name = "meta"
    model_save_name = "meta_model"
    log_dir = os.path.join(LOG_ROOT, lesson_name)
    model_dir = os.path.join(MODEL_ROOT, lesson_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    MODEL_SAVE_NAME = f"{model_save_name}.zip"
    model_path = os.path.join(model_dir, MODEL_SAVE_NAME)
    VEC_NORMALIZE_PATH = os.path.join(model_dir, "vec_normalize.pkl")
    
    # (NEW) Create the single, shared LocationMemory instance
    location_memory = LocationMemory(save_path=os.path.join(log_dir, 'location_memory.pkl'))
    location_memory.load_memory()

    # (NEW) Save hyperparameters to JSON
    hyperparameters = {
        "seed": seed, "n_epochs": n_epochs, "ent_coef": ent_coef, "gamma": gamma,
        "lesson": lesson_name, "model_name": model_save_name, "debug": args.debug,
        "num_cpu_arg": args.num_cpu, "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "local_num_cpu": None, "run_headless": None,
    }

    print(f"--- Starting training for: META-MANAGER AI (CEO) ---")
    print(f"--- Model will be saved to: {model_path} ---")

    if args.num_cpu:
        local_num_cpu = args.num_cpu
    else:
        auto_cores = max(1, os.cpu_count() - 2)
        print(f"--- Auto-detecting CPUs: Using {auto_cores} cores ---")
        local_num_cpu = auto_cores

    total_timesteps = args.total_timesteps
    n_steps = args.n_steps
    batch_size = args.batch_size
    run_headless = True
    
    if args.debug:
        print(f"--- DEBUG MODE: 1 CPU, VISIBLE window ---")
        local_num_cpu = 1
        run_headless = False
        
    # (NEW) Update hyperparameters with derived values
    hyperparameters["local_num_cpu"] = local_num_cpu
    hyperparameters["run_headless"] = run_headless
    hyperparam_filepath = os.path.join(log_dir, "hyperparameters.json")
    with open(hyperparam_filepath, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to: {hyperparam_filepath}")

    # (NEW) Pass location_memory to each env function
    env_fns = [make_env_fn(i, seed=seed, headless=run_headless, location_memory=location_memory) for i in range(local_num_cpu)]
    
    vec_env = DummyVecEnv(env_fns) if local_num_cpu == 1 else SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, log_dir)
    
    loading_model = os.path.exists(model_path) and os.path.exists(VEC_NORMALIZE_PATH)
    
    if loading_model:
        print(f"--- Found existing model and normalization stats, loading... ---")
        env = VecNormalize.load(VEC_NORMALIZE_PATH, vec_env)
        env.training = True 
    else:
        print(f"--- Creating new model and normalization stats... ---")
        env = VecNormalize(
            vec_env, 
            norm_obs=True, 
            norm_reward=True, 
            gamma=gamma, 
            norm_obs_keys=NORM_OBS_KEYS
        )

    device = "cuda" if th.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = th.cuda.get_device_name(0)
        print(f"Using device: {device} (Detected: {gpu_name})")
    else:
        print(f"Using device: {device}")

    print(f"Hyperparameters: n_steps={n_steps}, batch_size={batch_size}, num_cpu={local_num_cpu}")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 1000 // local_num_cpu), 
        save_path=model_dir, 
        name_prefix=f"{model_save_name}_checkpoint"
    )

    model = None 
    if loading_model:
        print(f"--- Loading and continuing training... ---")
        model = RecurrentPPOWithProgress.load(
            model_path, env=env, device=device, tensorboard_log=log_dir,
            n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
            gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs
        )
    else:
        print(f"--- Creating a new Meta-Manager model... ---")
        model = RecurrentPPOWithProgress(
            "MultiInputLstmPolicy",
            env, verbose=0, tensorboard_log=log_dir, device=device,
            n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
            gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs,
        )

    tb_writer = SummaryWriter(log_dir)
    logging_cb = LoggingCallback(writer=tb_writer, model_save_dir=model_dir, save_attention_every=100)
    progress_callback = ProgressBarCallback()
    all_callbacks = [checkpoint_callback, logging_cb, progress_callback]

    print("--- STARTING META-MANAGER AI (CEO) TRAINING ---")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=all_callbacks, 
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print(f"Saving final model to: {model_path}")
        if 'model' in locals():
            model.save(model_path)
        if 'env' in locals():
            try: 
                env.save(VEC_NORMALIZE_PATH)
                env.close()
            except Exception: pass
        if 'tb_writer' in locals():
            try: tb_writer.close()
            except Exception: pass
        # (NEW) Save the shared location memory
        if 'location_memory' in locals():
            location_memory.save_memory()
        print("Training complete.")

if __name__ == "__main__":
    main()
# train_specialist.py
# (FIXED: Added DebugDashboardWrapper and safe hyperparameters)

import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter 
import argparse
import json

from src.env.pokemon_env import PokemonEnv
# (NEW) Import the wrapper
from src.env.wrappers import TransposeObservationWrapper, InfoInjectorWrapper, DebugDashboardWrapper
from src.utils.callbacks import LoggingCallback
from src.agents.policy import CustomCombinedExtractor
from src.utils.location_memory import LocationMemory

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state") # Default state
LOG_ROOT = "logs"
MODEL_ROOT = "models"

# --- Hyperparameters ---
seed = 0
n_epochs = 20
ent_coef = 0.01
gamma = 0.997

NORM_OBS_KEYS = ["vision", "stats", "map", "menu_state", "items"]

# (ProgressBarCallback and RecurrentPPOWithProgress classes are unchanged)
class ProgressBarCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.locals['total_timesteps'], desc="Total Timesteps")
    def _on_step(self) -> bool:
        self.pbar.n = self.num_timesteps
        self.pbar.update(0)
        return True
    def _on_training_end(self) -> None:
        self.pbar.close()

class RecurrentPPOWithProgress(RecurrentPPO):
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        self.logger.record("train/ent_coef", self.ent_coef)
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        for epoch in tqdm(range(self.n_epochs), desc="Training on collected data...", leave=False):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(actions, th.Tensor):
                    actions = actions.long()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts
                )
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                if self.clip_range_vf is None:
                    values_pred = values.reshape(rollout_data.returns.shape)
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                else:
                    values_pred = values.reshape(rollout_data.returns.shape)
                    values_clipped = rollout_data.old_values + th.clamp(
                        values_pred - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    vf_loss1 = F.mse_loss(rollout_data.returns, values_pred)
                    vf_loss2 = F.mse_loss(rollout_data.returns, values_clipped)
                    value_loss = th.max(vf_loss1, vf_loss2)
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())
            self._n_updates += 1
            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                tqdm.write(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if entropy is not None:
            self.logger.record("train/entropy_loss", entropy_loss.item())
        if self.target_kl is not None:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))

# (NEW) Updated make_env_fn
def make_env_fn(rank, seed=0, headless=True, lesson="exploration", location_memory=None):
    """
    Utility function for creating a parallel environment instance.
    """
    def _init():
        state_file = STATE_PATH # Default
        
        env = PokemonEnv(
            rom_path=ROM_PATH,
            state_path=state_file, 
            headless=headless,
            lesson=lesson 
        )
        
        env = TransposeObservationWrapper(env)
        env = InfoInjectorWrapper(env, keys_to_inject=(
            "menu_state", "menu_stagnation_counter", 
            "stagnation_counter", "reward_trigger", "reward_triggers"
        ), location_memory=location_memory) 
        
        # (NEW) Add this if-statement
        if not headless:
            env = DebugDashboardWrapper(env)
        
        env.reset(seed=seed + rank)
        return env
        
    set_random_seed(seed)
    return _init

def main():
    """Main function to set up and run the training process."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesson', type=str, default='exploration', 
                        choices=['exploration', 'battle', 
                                 'healer', 'shopping', 'inventory', 'switch'],
                        help='Lesson to train on')
    parser.add_argument('--model_name', type=str, default='exploration_model', 
                        help='Specific name for the model file (e.g., exploration_model)')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in slow, visible debug mode (1 CPU, headless=False)')
    
    parser.add_argument('--num_cpu', type=int, default=None, 
                        help='Number of parallel environments (default: os.cpu_count() - 2)')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, 
                        help='Total timesteps to train for')
    
    # (FIXED) Lowered n_steps and batch_size to safe, standard values
    parser.add_argument('--n_steps', type=int, default=2048, 
                        help='Steps per env per update (rollout buffer size)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='PPO mini-batch size')
    
    args = parser.parse_args()

    lesson_name = args.lesson
    log_dir = os.path.join(LOG_ROOT, lesson_name)
    model_dir = os.path.join(MODEL_ROOT, lesson_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    MODEL_SAVE_NAME = f"{args.model_name}.zip"
    model_path = os.path.join(model_dir, MODEL_SAVE_NAME)
    
    VEC_NORMALIZE_PATH = os.path.join(model_dir, f"{args.model_name}_vec_normalize.pkl")

    location_memory = LocationMemory(save_path=os.path.join(log_dir, 'location_memory.pkl'))
    location_memory.load_memory()

    hyperparameters = {
        "seed": seed, "n_epochs": n_epochs, "ent_coef": ent_coef, "gamma": gamma,
        "lesson": args.lesson, "model_name": args.model_name, "debug": args.debug,
        "num_cpu_arg": args.num_cpu, "total_timesteps": args.total_timesteps,
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "local_num_cpu": None, "run_headless": None,
    }
    
    hyperparam_filepath = os.path.join(log_dir, "hyperparameters.json")
    with open(hyperparam_filepath, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to: {hyperparam_filepath}")

    print(f"--- Starting training for lesson: {args.lesson} ---")
    print(f"--- Model will be saved to: {model_path} ---")

    lesson_state_dir = os.path.join(STATE_DIR)
    
    if args.lesson != "exploration":
        lesson_files_wildcard = glob.glob(os.path.join(lesson_state_dir, f"{args.lesson}_*.state"))
        lesson_file_single = os.path.exists(os.path.join(lesson_state_dir, f"{args.lesson}.state"))
        
        if not lesson_files_wildcard and not lesson_file_single:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Lesson '{args.lesson}' requires '{args.lesson}.state' or '{args.lesson}_*.state'")
            print(f"files in the '{lesson_state_dir}' directory.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return

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

    with open(hyperparam_filepath, "r") as f:
        hyperparameters = json.load(f)
    hyperparameters["local_num_cpu"] = local_num_cpu
    hyperparameters["run_headless"] = run_headless
    with open(hyperparam_filepath, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Updated hyperparameters with derived values in: {hyperparam_filepath}")

    env_fns = [make_env_fn(i, seed=seed, headless=run_headless, lesson=args.lesson, location_memory=location_memory) for i in range(local_num_cpu)]
    
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
        print("Your RTX 4070 Ti should make this fast!")
    else:
        print(f"Using device: {device}")
    
    print(f"Hyperparameters: n_steps={n_steps}, batch_size={batch_size}, num_cpu={local_num_cpu}")

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // local_num_cpu),
        save_path=model_dir,
        name_prefix=f"{args.model_name}_checkpoint"
    )
    
    progress_callback = ProgressBarCallback()

    model = None 
    if loading_model:
        print(f"--- Found existing model at {model_path} ---")
        print("--- Loading and continuing training... ---")
        try:
            model = RecurrentPPOWithProgress.load(
                model_path, env=env, device=device, tensorboard_log=log_dir,
                n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
                gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs
            )
        except Exception as e:
            print(f"---!! WARNING: Failed to load model. !! ---")
            print(f"Error: {e}")
            print(f"--- Deleting old model/stats and creating new ones. ---")
            if os.path.exists(model_path): os.remove(model_path)
            if os.path.exists(VEC_NORMALIZE_PATH): os.remove(VEC_NORMALIZE_PATH)
            env.close() 
            vec_env = DummyVecEnv(env_fns) if local_num_cpu == 1 else SubprocVecEnv(env_fns)
            vec_env = VecMonitor(vec_env, log_dir)
            env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=True, 
                gamma=gamma, norm_obs_keys=NORM_OBS_KEYS
            )
            model = None 
        
    if model is None:
        print(f"--- Creating a new model... ---")
        model = RecurrentPPOWithProgress(
            "MultiInputLstmPolicy",
            env, verbose=0, tensorboard_log=log_dir, device=device,
            n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
            gamma=gamma, n_epochs=n_epochs, policy_kwargs=policy_kwargs,
        )

    tb_writer = SummaryWriter(log_dir)
    logging_cb = LoggingCallback(writer=tb_writer, model_save_dir=model_dir, save_attention_every=5_000)

    print("--- STARTING RECURRENT (LSTM) TRAINING ---")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, logging_cb, progress_callback], 
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        if 'model' in locals() and model is not None:
            print(f"Saving final model to: {model_path}")
            model.save(model_path)
        if 'env' in locals():
            try: 
                env.save(VEC_NORMALIZE_PATH)
                env.close()
            except Exception: pass
        if 'tb_writer' in locals():
            try: tb_writer.close()
            except Exception: pass
        if 'location_memory' in locals():
            location_memory.save_memory()
        print("Training complete.")

if __name__ == "__main__":
    main()
# src/utils/callbacks.py
# Refactored version of menu_logging.py
# (FIXED: Added logging for menu_state, script_state, 
# individual key_events, item_count, and exploration_map visual)

import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import torchvision.transforms as T
from PIL import Image

from src.utils.attention import generate_grad_cam

# Constants for indexing the 'stats' observation vector
STAT_PLAYER_X = 0
STAT_PLAYER_Y = 1
STAT_MAP_ID = 2
STAT_SCRIPT_STATE = 3 # (NEW) Added script state index
STAT_BADGES = 4
STAT_OPPONENT_HP = 5
STAT_OPPONENT_MAX_HP = 6
STAT_OPPONENT_LEVEL = 7
STAT_OPPONENT_SPECIES = 8
STAT_PARTY_START = 9
PARTY_POKEMON_SIZE = 4 # 4 stats per pokemon [HP, MaxHP, Level, Status]

# (NEW) Constants for item bag
ITEM_BAG_SIZE = 20
ITEM_SLOT_SIZE = 2

class LoggingCallback(BaseCallback):
    """
    A custom callback for logging detailed game state information, rewards,
    policy embeddings, and Grad-CAM attention maps to TensorBoard.
    
    This is the refactored version of 'MenuLoggingCallback'.
    """
    def __init__(self, writer, model_save_dir=".", verbose=0, save_attention_every=200_000):
        """
        Initializes the LoggingCallback.

        Args:
            writer: The TensorBoard SummaryWriter object.
            model_save_dir: Directory to save attention images.
            verbose: Verbosity level.
            save_attention_every: Frequency (in steps) to save attention images.
        """
        super().__init__(verbose)
        self.writer = writer
        self.model_save_dir = model_save_dir
        self.save_attention_every = save_attention_every if save_attention_every else 1_000_000_000
        self._last_attention_step = 0
        self.attention_dir = os.path.join(self.model_save_dir, "attention")
        os.makedirs(self.attention_dir, exist_ok=True)
        
        # (NEW) Define key event names for better logging
        self.key_event_names = [
            "GOT_POKEDEX", "GOT_OAKS_PARCEL", "BEAT_BROCK",
            "BEAT_RIVAL_CERULEAN", "GOT_BIKE_VOUCHER", "GOT_SS_TICKET",
        ]

    def _on_step(self) -> bool:
        """
        Called after each step. Logs info, game state, embeddings, and attention.
        """
        # --- 1. Log Info Dictionary ---
        infos = self.locals.get("infos", None)
        if infos:
            for i, info in enumerate(infos):
                # Log standard scalar info
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        # This will log info/menu_stagnation_counter, info/stagnation_counter, etc.
                        self.writer.add_scalar(f"info/{k}", v, self.num_timesteps)
                
                # Log detailed reward components
                if "reward_triggers" in info:
                    for reward, name in info.get("reward_triggers", []):
                        self.writer.add_scalar(f"reward_env{i}/{name}", reward, self.num_timesteps)
                
                # Log game progress metrics from info
                # (Note: These are custom metrics from the old env, may not all be active)
                self.writer.add_scalar("progress/seen_maps", info.get("seen_maps", 0), self.num_timesteps)
                self.writer.add_scalar("progress/map_coverage", info.get("map_coverage", 0), self.num_timesteps)
                self.writer.add_scalar("progress/events", info.get("events", 0), self.num_timesteps)
                self.writer.add_scalar("progress/pokedex", info.get("pokedex", 0), self.num_timesteps)
                self.writer.add_scalar("progress/heal_count", info.get("heal_count", 0), self.num_timesteps)
                self.writer.add_scalar("progress/party_size", info.get("party_size", 0), self.num_timesteps)
                self.writer.add_scalar("progress/visit_count", info.get("visit_count", 0), self.num_timesteps)
                self.writer.add_scalar("progress/money", info.get("money", 0), self.num_timesteps)
                self.writer.add_scalar("progress/badges", info.get("badges", 0), self.num_timesteps)
                self.writer.add_scalar("progress/seen_pokemons", info.get("seen_pokemons", 0), self.num_timesteps)
                self.writer.add_scalar("progress/caught_pokemons", info.get("caught_pokemons", 0), self.num_timesteps)

        obs_data = self.locals.get("new_obs", None) or self.locals.get("obs", None)
        
        if hasattr(self.training_env, "get_original_obs"):
            new_obs = self.training_env.get_original_obs()
        else:
            new_obs = obs_data


        # --- 2. Log Game State (from Observation) ---
        if new_obs:
            # Sample observation from the first environment
            obs_sample = {k: v[0] for k, v in new_obs.items()}

            # Log player/map state
            self.writer.add_scalar("game_state/player_x", obs_sample["stats"][STAT_PLAYER_X], self.num_timesteps)
            self.writer.add_scalar("game_state/player_y", obs_sample["stats"][STAT_PLAYER_Y], self.num_timesteps)
            self.writer.add_scalar("game_state/map_id", obs_sample["stats"][STAT_MAP_ID], self.num_timesteps)
            self.writer.add_scalar("game_state/badges", obs_sample["stats"][STAT_BADGES], self.num_timesteps)
            
            # (NEW) Log missing critical stats
            self.writer.add_scalar("game_state/script_state", obs_sample["stats"][STAT_SCRIPT_STATE], self.num_timesteps)
            self.writer.add_scalar("game_state/current_menu", obs_sample["menu_state"][0], self.num_timesteps)
            self.writer.add_scalar("game_state/menu_cursor", obs_sample["menu_state"][1], self.num_timesteps)

            # Log battle state
            self.writer.add_scalar("battle/opponent_hp", obs_sample["stats"][STAT_OPPONENT_HP], self.num_timesteps)
            self.writer.add_scalar("battle/opponent_max_hp", obs_sample["stats"][STAT_OPPONENT_MAX_HP], self.num_timesteps)
            self.writer.add_scalar("battle/opponent_level", obs_sample["stats"][STAT_OPPONENT_LEVEL], self.num_timesteps)
            self.writer.add_scalar("battle/opponent_species", obs_sample["stats"][STAT_OPPONENT_SPECIES], self.num_timesteps)

            # Log party stats
            party_stats = obs_sample["stats"][STAT_PARTY_START:]
            party_count = 0
            for i in range(6):
                hp_idx = i * PARTY_POKEMON_SIZE
                hp = party_stats[hp_idx]
                max_hp = party_stats[hp_idx + 1]
                level = party_stats[hp_idx + 2]
                status = party_stats[hp_idx + 3]

                if max_hp > 0: # Only count/log if pokemon exists
                    party_count += 1
                
                self.writer.add_scalar(f"party/pokemon_{i+1}_hp", hp, self.num_timesteps)
                self.writer.add_scalar(f"party/pokemon_{i+1}_max_hp", max_hp, self.num_timesteps)
                self.writer.add_scalar(f"party/pokemon_{i+1}_level", level, self.num_timesteps)
                self.writer.add_scalar(f"party/pokemon_{i+1}_status", status, self.num_timesteps)
            
            self.writer.add_scalar("game_state/party_count", party_count, self.num_timesteps)

            # Log key events
            key_events = obs_sample["key_events"]
            self.writer.add_scalar("game_state/key_events_sum", sum(key_events), self.num_timesteps)
            
            # (NEW) Log individual key events
            for i, event_val in enumerate(key_events):
                event_name = self.key_event_names[i] if i < len(self.key_event_names) else f"event_{i}"
                self.writer.add_scalar(f"key_events/{event_name}", event_val, self.num_timesteps)
                
            # (NEW) Log item count
            item_ids = obs_sample["items"][::ITEM_SLOT_SIZE] # Get all item IDs
            total_item_count = np.count_nonzero(item_ids)
            self.writer.add_scalar("game_state/total_item_count", total_item_count, self.num_timesteps)
            
            # (NEW) Log exploration map visual
            map_img = obs_sample["map"] # This is (H, W)
            self.writer.add_image("visuals/exploration_map", map_img, self.num_timesteps, dataformats='HW')


        # --- 3. Log Policy Embeddings ---
        try:
            if obs_data:
                max_samples = min(32, obs_data[next(iter(obs_data))].shape[0])
                batch_tensors = self._prepare_obs_batch(obs_data, max_samples=max_samples)
                
                with th.no_grad():
                    features = self.model.policy.features_extractor(batch_tensors)
                    feat_np = features.cpu().numpy()
                    self.writer.add_histogram("policy/embeddings", feat_np, self.num_timesteps)
        except Exception as e:
            print(f"LoggingCallback: failed to log embeddings: {e}")

        # --- 4. Log Attention Images (Grad-CAM) ---
        if self.save_attention_every and (self.num_timesteps - self._last_attention_step) >= self.save_attention_every:
            if new_obs:
                self._save_attention_images(new_obs)
            self._last_attention_step = self.num_timesteps
        
        return True

    def _prepare_obs_batch(self, obs_dict, max_samples=16):
        """
        Prepares a small, randomized batch of observations for embedding extraction.
        Handles device transfer and vision data normalization.
        """
        device = self.model.device
        batch = {}
        first_key = next(iter(obs_dict))
        n_envs = obs_dict[first_key].shape[0]
        if n_envs == 0:
            raise ValueError("No envs found in obs.")
            
        idx = np.random.choice(n_envs, size=min(max_samples, n_envs), replace=False)
        
        for k, v in obs_dict.items():
            sel = v[idx]
            t = th.tensor(sel).to(device)
            
            if k == "vision" and t.dtype == th.uint8:
                t = t.float() / 255.0
            elif t.dtype != th.float32:
                t = t.float()
            
            batch[k] = t
        return batch

    def _save_attention_images(self, original_obs):
        """
        Generates and saves Grad-CAM attention images for a few sample envs.
        """
        if not original_obs:
            return
            
        n_envs = original_obs[next(iter(original_obs))].shape[0]
        indices = list(range(min(4, n_envs))) # Sample up to 4 envs
        
        for i in indices:
            single_obs = {k: v[i] for k, v in original_obs.items()}
            
            try:
                img_overlay = generate_grad_cam(self.model, single_obs, device=self.model.device)
                step = self.num_timesteps
                path = os.path.join(self.attention_dir, f"attention_step_{step}_env{i}.png")
                
                Image.fromarray(img_overlay).save(path)
                
                to_tensor = T.ToTensor()
                img_t = to_tensor(img_overlay).unsqueeze(0)
                self.writer.add_image(f"attention/env{i}", img_t[0], step)
                
            except Exception as e:
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"LoggingCallback: FAILED to create attention image for env {i}")
                print(f"ERROR: {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    def _on_training_end(self) -> None:
        try:
            self.writer.flush()
        except Exception:
            pass
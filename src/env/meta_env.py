# src/env/meta_env.py
# (FIXED: Now accepts a LocationMemory object to pass to ManagerEnv)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from sb3_contrib import RecurrentPPO

from src.env.manager_env import ManagerEnv, NUM_GOALS, KEY_EVENT_NAMES
from src.env.wrappers import TransposeObservationWrapper
from src.agents.policy import CustomCombinedExtractor 
from src.utils.location_memory import LocationMemory # (NEW) Import

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")
MODEL_ROOT = "models"
MANAGER_MODEL_PATH = os.path.join(MODEL_ROOT, "manager", "manager_model.zip")

class MetaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # (FIXED) Added location_memory=None
    def __init__(self, rom_path=ROM_PATH, state_path=STATE_PATH, headless=True, location_memory=None):
        super().__init__()

        print("--- Initializing Meta-Manager Environment ---")
        
        # (FIXED) Pass location_memory to the ManagerEnv
        self.manager_env = ManagerEnv(
            rom_path=rom_path,
            state_path=state_path,
            headless=headless,
            location_memory=location_memory 
        )

        if not os.path.exists(MANAGER_MODEL_PATH):
            raise FileNotFoundError(f"Trained Manager model not found at {MANAGER_MODEL_PATH}. Please train the Manager first.")
            
        print(f"Loading trained Manager model from {MANAGER_MODEL_PATH}...")
        self.manager_model = RecurrentPPO.load(MANAGER_MODEL_PATH, device="cpu")
        print("Manager model loaded.")

        self.observation_space = self.manager_env.observation_space
        self.action_space = spaces.Discrete(NUM_GOALS)
        
        self.key_event_names = KEY_EVENT_NAMES
        self.current_meta_obs = None
        self.last_info = {}
        print("--- Meta-Manager Environment Initialized ---")

    def reset(self, seed=None, options=None):
        obs, info = self.manager_env.reset(seed=seed)
        self.current_meta_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action):
        goal_index = action.item()
        self.manager_env.set_goal(goal_index)
        goal_name = self.key_event_names[goal_index]
        
        manager_lstm_states = None
        total_reward = 0.0

        while True:
            manager_action, manager_lstm_states = self.manager_model.predict(
                self.current_meta_obs,
                state=manager_lstm_states,
                deterministic=True
            )
            
            obs, reward, terminated, truncated, info = self.manager_env.step(manager_action)
            
            self.current_meta_obs = obs
            self.last_info = info
            total_reward += reward 
            
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

    def close(self):
        self.manager_env.close()
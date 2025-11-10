# src/env/manager_env.py
# (FIXED: Set lesson to "manager" to receive all reward types)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from sb3_contrib import RecurrentPPO

from src.env.pokemon_env import PokemonEnv
from src.env.wrappers import TransposeObservationWrapper, InfoInjectorWrapper
from src.utils.location_memory import LocationMemory

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")
MODEL_ROOT = "models"

# Define the goals the Meta-Manager can choose from.
KEY_EVENT_NAMES = [
    "GOT_POKEDEX", "GOT_OAKS_PARCEL", "BEAT_BROCK",
    "BEAT_RIVAL_CERULEAN", "GOT_BIKE_VOUCHER", "GOT_SS_TICKET",
]
NUM_GOALS = len(KEY_EVENT_NAMES)

SPECIALIST_NAMES = [
    "explore", "battle", "healer", "shopping",
    "inventory", "switch", "go_to_memorable_location"
]

MANAGER_STEP_SIZE = 200

class ManagerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rom_path=ROM_PATH, state_path=STATE_PATH, headless=True):
        super().__init__()

        print("--- Initializing Manager Environment ---")
        self.location_memory = LocationMemory()
        
        # (FIXED) Set lesson to "manager" to receive all reward signals
        self.base_env = PokemonEnv(
            rom_path=rom_path,
            state_path=state_path,
            headless=headless,
            lesson="manager", 
            location_memory=self.location_memory
        )
        
        self.base_env = TransposeObservationWrapper(self.base_env)
        self.base_env = InfoInjectorWrapper(self.base_env)

        self.observation_space = self.base_env.observation_space
        self.action_space = spaces.Discrete(len(SPECIALIST_NAMES))

        self.specialists = {}
        self.specialist_lstm_states = {}

        print("--- Loading Specialist Models for Manager ---")
        for name in SPECIALIST_NAMES:
            model_path = os.path.join(MODEL_ROOT, name, f"{name}_model.zip")
            if os.path.exists(model_path):
                print(f"Loading specialist: {name}...")
                self.specialists[name] = RecurrentPPO.load(model_path, env=self.base_env, device="cpu")
                self.specialist_lstm_states[name] = None
            else:
                print(f"WARNING: Specialist model '{name}' not found at {model_path}.")
        
        if "explore" not in self.specialists:
            raise FileNotFoundError("Critical Error: The 'explore' specialist is required.")

        self.current_base_obs = None
        self.last_info = {}
        self.SPECIALIST_NAMES = SPECIALIST_NAMES
        
        self.key_event_names = KEY_EVENT_NAMES
        self.current_goal_index = None
        self.last_key_events = np.zeros(NUM_GOALS, dtype=np.uint8)
        
        print("--- (GOAL-ORIENTED) Manager Environment Initialized ---")

    def set_goal(self, goal_index: int):
        if goal_index >= NUM_GOALS:
            raise ValueError(f"Invalid goal index: {goal_index}")
        self.current_goal_index = goal_index
        goal_name = self.key_event_names[goal_index]
        print(f"--- MANAGER ENV: New Goal Set: {goal_name} (Index {goal_index}) ---")

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed)
        
        for name in self.specialists:
            self.specialist_lstm_states[name] = None
            
        self.current_base_obs = obs
        self.last_info = info
        
        if self.current_goal_index is None:
            self.set_goal(0) 
            
        self.last_key_events = obs["key_events"]
        return obs, info

    def step(self, action):
        specialist_name = self.SPECIALIST_NAMES[action]
        
        if specialist_name == "go_to_memorable_location":
            if self.location_memory and len(self.location_memory) > 0:
                target_coords = self.location_memory.get_prioritized_locations()[0]
                self.base_env.set_target_coords(target_coords)
                specialist_name = "explore"
            else:
                return self.current_base_obs, -0.1, False, False, self.last_info 

        if specialist_name not in self.specialists:
            return self.current_base_obs, -0.5, False, False, self.last_info 

        model = self.specialists[specialist_name]
        lstm_state = self.specialist_lstm_states[specialist_name]
        current_obs = self.current_base_obs

        for _ in range(MANAGER_STEP_SIZE):
            specialist_action, lstm_state = model.predict(
                current_obs,
                state=lstm_state,
                deterministic=True
            )
            
            next_obs, base_reward, terminated, truncated, info = self.base_env.step(specialist_action.item())
            
            current_obs = next_obs
            self.last_info = info
            
            current_key_events = next_obs["key_events"]
            goal_achieved = False
            
            if (current_key_events[self.current_goal_index] == 1 and 
                self.last_key_events[self.current_goal_index] == 0):
                goal_achieved = True
            
            self.last_key_events = current_key_events
            
            if goal_achieved:
                print(f"!!! GOAL ACHIEVED: {self.key_event_names[self.current_goal_index]} !!!")
                return current_obs, 100.0, True, truncated, info

            if terminated or truncated:
                print("ManagerEnv: Base episode finished (goal not met).")
                for name in self.specialists:
                    self.specialist_lstm_states[name] = None
                self.current_base_obs = current_obs
                return current_obs, 0.0, terminated, truncated, info

        self.specialist_lstm_states[specialist_name] = lstm_state
        self.current_base_obs = current_obs
        
        return self.current_base_obs, 0.0, False, False, self.last_info

    def close(self):
        self.base_env.close()
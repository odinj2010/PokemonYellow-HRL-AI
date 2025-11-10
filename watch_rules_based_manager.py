# watch_rules_based_manager.py
# This is the refactored version of 'watch_galaxy_ai.py'
# It runs a hard-coded, rules-based "Manager" that switches between
# specialist models based on the current game state.

import time
from sb3_contrib import RecurrentPPO
import os
import gymnasium as gym
import numpy as np

# (NEW) Refactored imports from the 'src' directory structure
from src.env.pokemon_env import PokemonEnv
from src.env.wrappers import TransposeObservationWrapper, DebugDashboardWrapper # (NEW)

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")
MODEL_ROOT = "models" 

# (NEW) Updated model paths to use 'battle' instead of 'brock'
MODEL_PATHS = {
    "explore": os.path.join(MODEL_ROOT, "exploration", "exploration_model.zip"),
    "battle": os.path.join(MODEL_ROOT, "battle", "battle_model.zip"),
    "healer": os.path.join(MODEL_ROOT, "healer", "healer_model.zip"),
    "shopping": os.path.join(MODEL_ROOT, "shopping", "shopping_model.zip"),
    "inventory": os.path.join(MODEL_ROOT, "inventory", "inventory_model.zip"),
    "switch": os.path.join(MODEL_ROOT, "switch", "switch_model.zip"),
}

# --- Memory and Observation Index Constants ---
ADDR_BATTLE_TYPE = 0xD05A
ADDR_CURRENT_MENU = 0xD057
MENU_OVERWORLD = 0x00

STAT_MAP_ID = 2
STAT_SCRIPT_STATE = 3
STAT_PARTY_START = 9
PARTY_POKEMON_SIZE = 4
STAT_HP_OFFSET = 0
STAT_MAX_HP_OFFSET = 1

LOW_HP_THRESHOLD = 0.6
CRITICAL_HP_THRESHOLD = 0.3

# (FIXED) --- Corrected Map ID for Viridian Mart ---
MAP_ID_VIRIDIAN_MART = 42 


# --- Environment Setup ---
print("Creating visible quest environment...")
env = PokemonEnv(
    rom_path=ROM_PATH, 
    state_path=STATE_PATH,
    headless=False,
    lesson="manager" # Use "manager" lesson to get all rewards
)
env = TransposeObservationWrapper(env)
env = DebugDashboardWrapper(env) # (NEW) Add this line
models = {}

# --- Model Loading ---
print("--- Loading Rules-Based Manager Models ---")
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        print(f"Loading model: {name}...")
        try:
            models[name] = RecurrentPPO.load(path, env=env, device="cpu")
        except Exception as e:
            print(f"---!! WARNING: Failed to load model '{name}'. !! ---")
            print(f"Error: {e}")
    else:
        print(f"Warning: Model '{name}' not found at {path}. AI will be handicapped.")
        
if "explore" not in models:
    print(f"Error: The default 'explore' model is required. Please train it first.")
    exit()
    
print("--- All available models loaded. Rules-Based Manager is ONLINE. ---")

# --- Initial State Setup ---
obs, info = env.reset()
lstm_states = {name: None for name in models.keys()}
current_model_name = "explore"
pyboy_env = env.unwrapped

def get_party_hp_string(stats_vec):
    """
    Helper function to format the HP of all party Pokémon into a string.
    (No longer used by the main print loop, but kept for potential debugging)
    """
    party_hps = []
    for i in range(6):
        hp_idx = STAT_PARTY_START + (i * PARTY_POKEMON_SIZE) + STAT_HP_OFFSET
        max_hp_idx = STAT_PARTY_START + (i * PARTY_POKEMON_SIZE) + STAT_MAX_HP_OFFSET
        
        hp = stats_vec[hp_idx]
        max_hp = stats_vec[max_hp_idx]
        
        if max_hp > 0:
            party_hps.append(f"P{i+1}: {hp}/{max_hp}")
    return " | ".join(party_hps)

# --- Main Watch Loop ---
while True:
    try:
        # --- 1. Read State for Decision Making ---
        battle_type = pyboy_env.pyboy.memory[ADDR_BATTLE_TYPE]
        menu_type = pyboy_env.pyboy.memory[ADDR_CURRENT_MENU]
        stats = obs["stats"].squeeze()
        map_id = stats[STAT_MAP_ID]
        script_state = stats[STAT_SCRIPT_STATE]
        party_count = pyboy_env.pyboy.memory[0xD163]
        
        is_party_injured = False
        active_pkmn_hp = 1
        active_pkmn_max_hp = 1
        
        for i in range(party_count):
            hp_idx = STAT_PARTY_START + (i * PARTY_POKEMON_SIZE) + STAT_HP_OFFSET
            max_hp_idx = STAT_PARTY_START + (i * PARTY_POKEMON_SIZE) + STAT_MAX_HP_OFFSET
            hp = stats[hp_idx]
            max_hp = stats[max_hp_idx]
            
            if i == 0:
                active_pkmn_hp = hp
                active_pkmn_max_hp = max(1, max_hp)
            
            if max_hp > 0 and hp < max_hp:
                is_party_injured = True
                
        hp_ratio = active_pkmn_hp / active_pkmn_max_hp
        new_model_name = "explore"
        
        # --- 2. Hierarchical Decision Logic (Rules-Based Policy) ---
        
        if battle_type > 0:
            # **IN BATTLE**
            if (hp_ratio < CRITICAL_HP_THRESHOLD and "switch" in models and party_count > 1):
                new_model_name = "switch"
            elif (hp_ratio < LOW_HP_THRESHOLD and "inventory" in models):
                new_model_name = "inventory"
            elif "battle" in models:
                new_model_name = "battle"
            else:
                new_model_name = "explore" # Fallback if no battle model
                
        elif menu_type != MENU_OVERWORLD or script_state != 0: 
            # **IN MENU/SCRIPT**
            if map_id == MAP_ID_VIRIDIAN_MART and "shopping" in models:
                new_model_name = "shopping"
            else:
                new_model_name = "explore" # Fallback to get out of menus
        
        else:
            # **OVERWORLD**
            if (is_party_injured and "healer" in models):
                new_model_name = "healer"
            else:
                new_model_name = "explore"
                
        # Final check
        if new_model_name not in models:
            current_model_name = "explore"
        else:
            current_model_name = new_model_name
            
    except Exception as e:
        print(f"Warning: Could not read memory. Sticking with {current_model_name}. Error: {e}")
        
    # --- 3. Specialist Execution ---
    model_to_use = models[current_model_name]
    state_to_use = lstm_states[current_model_name]
    
    action, new_state = model_to_use.predict(
        obs, 
        state=state_to_use, 
        deterministic=True
    )
    
    lstm_states[current_model_name] = new_state
    obs, reward, terminated, truncated, info = env.step(action.item())
    
    # --- 4. Logging ---
    # The debug dashboard now shows all the important info.
    # We just print the active model name.
    print(f"\rModel: {current_model_name.upper():<10}   ", end="")
    
    # --- 5. Episode Reset ---
    if terminated or truncated:
        print("\nEpisode finished. Resetting...")
        obs, info = env.reset()
        lstm_states = {name: None for name in models.keys()}
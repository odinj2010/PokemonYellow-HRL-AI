# src/env/pokemon_env.py
# (FINAL VERSION: Confirmed coordinates and optimized rewards applied)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent 
import collections
from PIL import Image
import os
import glob 
import random 
from pathlib import Path

# --- Create an absolute path to the project root ---
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(SCRIPT_DIR, "states")

# --- Action Maps (Unchanged) ---
ACTION_PRESS_MAP = {
    0: WindowEvent.PRESS_ARROW_UP,
    1: WindowEvent.PRESS_ARROW_DOWN,
    2: WindowEvent.PRESS_ARROW_LEFT,
    3: WindowEvent.PRESS_ARROW_RIGHT,
    4: WindowEvent.PRESS_BUTTON_A,      # 'A' button
    5: WindowEvent.PRESS_BUTTON_B,      # 'B' button
    6: WindowEvent.PRESS_BUTTON_START,  # 'Start' button
    7: None,                            # 'No-Op' (do nothing)
}

ACTION_RELEASE_MAP = {
    0: WindowEvent.RELEASE_ARROW_UP,
    1: WindowEvent.RELEASE_ARROW_DOWN,
    2: WindowEvent.RELEASE_ARROW_LEFT,
    3: WindowEvent.RELEASE_ARROW_RIGHT,
    4: WindowEvent.RELEASE_BUTTON_A,
    5: WindowEvent.RELEASE_BUTTON_B,
    6: WindowEvent.RELEASE_BUTTON_START,
    7: None,
}

# --- Memory Addresses ---
ADDR_CURRENT_MENU = 0xD057
ADDR_MENU_CURSOR_POS = 0xCC25
ADDR_SCRIPT_STATE = 0xC506
ADDR_BATTLE_TYPE = 0xD05A
ADDR_BADGES = 0xD356
ADDR_MONEY_HI = 0xD347
ADDR_MONEY_MID = 0xD348
ADDR_MONEY_LO = 0xD349
ADDR_NURSE_JOY_SCRIPT = 0xC660
ADDR_TEXT_BUFFER_START = 0xC5B0
ADDR_TEXT_BUFFER_END = 0xC6B0 

# --- (CONFIRMED) ADDRESSES BASED DEBUGGING ---
ADDR_MAP_ID = 0xD35D
ADDR_PLAYER_Y = 0xD360
ADDR_PLAYER_X = 0xD361

# --- Text Hashes ---
HASH_SUPER_EFFECTIVE = 9181516134342498431
HASH_NOT_VERY_EFFECTIVE = 6822968668854900320
HASH_NO_EFFECT = {-7197159403776133141, -9144707337852517228}

# --- Key Event Flags ---
ADDR_MAIN_EVENT_FLAGS_1 = 0xD730
KEY_EVENT_FLAGS_DEF = {
    "GOT_POKEDEX": (0xD730, 0),
    "GOT_OAKS_PARCEL": (0xD730, 4),
    "BEAT_BROCK": (0xD730, 7),
    "BEAT_RIVAL_CERULEAN": (0xD731, 2),
    "GOT_BIKE_VOUCHER": (0xD732, 0),
    "GOT_SS_TICKET": (0xD732, 1),
}
NUM_KEY_EVENTS = len(KEY_EVENT_FLAGS_DEF)

# --- Item Bag (Unchanged) ---
ADDR_ITEM_BAG_START = 0xD31D
ITEM_BAG_SIZE = 20
ITEM_SLOT_SIZE = 2
ITEM_ID_ANTIDOTE = 0x0E
ITEM_ID_PARLYZ_HEAL = 0x0C

# --- Pokedex (Unchanged) ---
ADDR_POKEDEX_CAUGHT = 0xD2F7

# --- Opponent Battle Data (Unchanged) ---
ADDR_OPPONENT_SPECIES = 0xCFF1
ADDR_OPPONENT_HP_HI = 0xCFF4
ADDR_OPPONENT_HP_LO = 0xCFF5
ADDR_OPPONENT_STATUS = 0xCFF7
ADDR_OPPONENT_MAX_HP_HI = 0xCFF8
ADDR_OPPONENT_MAX_HP_LO = 0xCFF9
ADDR_OPPONENT_LEVEL = 0xCFFD

# --- Party Data (Unchanged) ---
PARTY_DATA_START = 0xD16A
PARTY_MEMBER_SIZE = 44
OFFSET_SPECIES = 0x00
OFFSET_STATUS = 0x05
STATUS_POISON = 0x08
STATUS_PARALYSIS = 0x40
OFFSET_HP_HI = 0x06
OFFSET_HP_LO = 0x07
OFFSET_LEVEL = 0x1B
OFFSET_MAX_HP_HI = 0x1C
OFFSET_MAX_HP_LO = 0x1D
OFFSET_EXP_HI = 0x0D
OFFSET_EXP_MID = 0x0E
OFFSET_EXP_LO = 0x0F

# --- Constants for Environment Configuration (ADJUSTED) ---
VISION_GRID_SIZE = 84
FRAME_STACK = 4
EXPLORATION_MAP_SIZE = 128
MENU_STAGNATION_THRESHOLD = 100
STAGNATION_THRESHOLD = 100
STAGNATION_PENALTY = -2.5 # Adjusted from -1.0 to -2.5
TIME_PENALTY = -0.01

class PokemonEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rom_path, state_path, headless=False, lesson="exploration", location_memory=None):
        super().__init__()
        self.lesson = lesson
        self.default_state_path = state_path
        self.location_memory = location_memory
        if self.location_memory:
            self.location_memory.load_memory()
            Path('logs/memory_images').mkdir(parents=True, exist_ok=True)
        
        self.state_paths = []
        if self.lesson not in ["exploration", "manager"]:
            self.state_paths = glob.glob(os.path.join(STATE_DIR, f"{self.lesson}_*.state"))
            fallback_path = os.path.join(STATE_DIR, f"{self.lesson}.state")
            if not self.state_paths and os.path.exists(fallback_path):
                self.state_paths = [fallback_path]
        
        if not self.state_paths:
            self.state_paths = [self.default_state_path]
            
        print(f"--- {self.lesson.upper()} LESSON: Found {len(self.state_paths)} state file(s) ---")

        self.headless = headless
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window)
        
        if headless:
            self.pyboy.set_emulation_speed(0)
        else:
            self.pyboy.set_emulation_speed(1) # Changed from 4 to 1 for stability

        self.action_space = spaces.Discrete(8) # 8 actions

        # Observation space definition
        party_stats_low = []
        party_stats_high = []
        for _ in range(6):
            party_stats_low.extend([0, 0, 0, 0])
            party_stats_high.extend([65535, 65535, 100, 255])
        stats_space_low = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0] + party_stats_low, dtype=np.uint16
        )
        stats_space_high = np.array(
            [255, 255, 255, 255, 255, 65535, 65535, 100, 255] + party_stats_high, dtype=np.uint16
        )
        stats_shape = (stats_space_low.shape[0],)
        obs_space = {
            "vision": spaces.Box(
                low=0, high=255,
                shape=(VISION_GRID_SIZE, VISION_GRID_SIZE, FRAME_STACK),
                dtype=np.uint8
            ),
            "stats": spaces.Box(
                low=stats_space_low, high=stats_space_high,
                shape=stats_shape,
                dtype=np.uint16
            ),
            "key_events": spaces.MultiBinary(NUM_KEY_EVENTS),
            "map": spaces.Box(
                low=0, high=255,
                shape=(EXPLORATION_MAP_SIZE, EXPLORATION_MAP_SIZE),
                dtype=np.uint8
            ),
            "menu_state": spaces.Box(
                low=0, high=255, shape=(2,), dtype=np.uint8
            ),
            "items": spaces.Box(
                low=0, high=255,
                shape=(ITEM_BAG_SIZE * ITEM_SLOT_SIZE,),
                dtype=np.uint8
            ),
        }
        if self.location_memory:
            obs_space["memorable_location"] = spaces.Box(
                low=0, high=255, shape=(3,), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_space)

        # Internal State Variables
        self.frame_stack = collections.deque(maxlen=FRAME_STACK)
        self.visited_tiles = set()
        self.visited_maps = set()
        self.exploration_map = np.zeros((EXPLORATION_MAP_SIZE, EXPLORATION_MAP_SIZE), dtype=np.uint8)
        self.seen_dialogue_hashes = set()
        self.empty_dialogue_hash = hash(tuple([0] * (ADDR_TEXT_BUFFER_END - ADDR_TEXT_BUFFER_START)))
        self.quest_flags = {}
        self.menu_stagnation_counter = 0
        self.title_screen_map_id = 0
        self.step_count = 0
        self.max_steps = 10000
        self.last_action = 0
        self.stagnation_counter = 0
        self.last_coords = (0, 0, 0)
        self.last_reward_components = []
        self.target_coords = None

    def set_target_coords(self, coords):
        self.target_coords = coords

    def _bcd_to_int(self, bcd_bytes):
        """Converts a list/array of BCD bytes to an integer."""
        val = 0
        try:
            for byte in bcd_bytes:
                hi = (byte >> 4) & 0x0F
                lo = byte & 0x0F
                val = (val * 100) + (hi * 10) + lo
        except Exception as e:
            print(f"Error converting BCD: {e}, bytes: {bcd_bytes}")
            return 0
        return val

    def _read_party_stats(self):
        party_stats = []
        party_levels = []
        party_hp = []
        party_max_hp = []
        party_status = []
        party_exp = []
        party_species = []
        party_count = self.pyboy.memory[0xD163]
        for i in range(6):
            if i < party_count:
                base_addr = PARTY_DATA_START + (i * PARTY_MEMBER_SIZE)
                hp = (self.pyboy.memory[base_addr + OFFSET_HP_HI] << 8) + self.pyboy.memory[base_addr + OFFSET_HP_LO]
                max_hp = (self.pyboy.memory[base_addr + OFFSET_MAX_HP_HI] << 8) + self.pyboy.memory[base_addr + OFFSET_MAX_HP_LO]
                level = self.pyboy.memory[base_addr + OFFSET_LEVEL]
                status = self.pyboy.memory[base_addr + OFFSET_STATUS]
                exp = (self.pyboy.memory[base_addr + OFFSET_EXP_HI] << 16) + (self.pyboy.memory[base_addr + OFFSET_EXP_MID] << 8) + self.pyboy.memory[base_addr + OFFSET_EXP_LO]
                species = self.pyboy.memory[base_addr + OFFSET_SPECIES]
                party_stats.extend([hp, max_hp, level, status])
                party_levels.append(level)
                party_hp.append(hp)
                party_max_hp.append(max_hp)
                party_status.append(status)
                party_exp.append(exp)
                party_species.append(species)
            else:
                party_stats.extend([0, 0, 0, 0])
                party_levels.append(0)
                party_hp.append(0)
                party_max_hp.append(0)
                party_status.append(0)
                party_exp.append(0)
                party_species.append(0)
        return party_stats, party_levels, party_hp, party_max_hp, party_status, party_exp, party_species


    def _get_observation(self):
        raw_pixels = self.pyboy.screen.ndarray
        pil_image = Image.fromarray(raw_pixels, 'RGBA').convert('L')
        resized_image = pil_image.resize((VISION_GRID_SIZE, VISION_GRID_SIZE), Image.NEAREST)
        frame = np.array(resized_image, dtype=np.uint8)
        frame = np.expand_dims(frame, axis=-1)
        if len(self.frame_stack) == 0:
            for _ in range(FRAME_STACK):
                self.frame_stack.append(frame)
        else:
            self.frame_stack.append(frame)
        stacked_frames = np.concatenate(list(self.frame_stack), axis=-1)
        
        enemy_hp = (self.pyboy.memory[ADDR_OPPONENT_HP_HI] << 8) + self.pyboy.memory[ADDR_OPPONENT_HP_LO]
        enemy_max_hp = (self.pyboy.memory[ADDR_OPPONENT_MAX_HP_HI] << 8) + self.pyboy.memory[ADDR_OPPONENT_MAX_HP_LO]
        enemy_level = self.pyboy.memory[ADDR_OPPONENT_LEVEL]
        enemy_species = self.pyboy.memory[ADDR_OPPONENT_SPECIES]
        
        party_stats, party_levels, party_hp, party_max_hp, party_status, party_exp, party_species = self._read_party_stats()
        self.quest_flags["party_levels"] = party_levels
        self.quest_flags["party_hp"] = party_hp
        self.quest_flags["party_max_hp"] = party_max_hp 
        self.quest_flags["party_status"] = party_status
        self.quest_flags["party_exp"] = party_exp
        self.quest_flags["party_species"] = party_species
        
        # (FIXED) Read the new, confirmed addresses
        stats_obs = np.array([
            self.pyboy.memory[ADDR_PLAYER_X], self.pyboy.memory[ADDR_PLAYER_Y],
            self.pyboy.memory[ADDR_MAP_ID], self.pyboy.memory[ADDR_SCRIPT_STATE],
            self.pyboy.memory[ADDR_BADGES], enemy_hp, enemy_max_hp,
            enemy_level, enemy_species,
        ] + party_stats, dtype=np.uint16)
        
        key_events_obs = self._read_key_event_flags()
        
        current_map = stats_obs[2]
        player_x = stats_obs[0]
        player_y = stats_obs[1]
        
        clamped_x = max(0, min(player_x, EXPLORATION_MAP_SIZE - 1))
        clamped_y = max(0, min(player_y, EXPLORATION_MAP_SIZE - 1))
        self.exploration_map[clamped_y, clamped_x] = current_map

        menu_state_obs = np.array([
            self.pyboy.memory[ADDR_CURRENT_MENU],
            self.pyboy.memory[ADDR_MENU_CURSOR_POS],
        ], dtype=np.uint8)
        item_bag_data = self.pyboy.memory[ADDR_ITEM_BAG_START : ADDR_ITEM_BAG_START + (ITEM_BAG_SIZE * ITEM_SLOT_SIZE)]
        items_obs = np.array(item_bag_data, dtype=np.uint8)
        if len(items_obs) < (ITEM_BAG_SIZE * ITEM_SLOT_SIZE):
            items_obs = np.pad(items_obs, (0, (ITEM_BAG_SIZE * ITEM_SLOT_SIZE) - len(items_obs)), 'constant')
        
        obs_dict = {
            "vision": stacked_frames, "stats": stats_obs,
            "key_events": key_events_obs, "map": self.exploration_map,
            "menu_state": menu_state_obs, "items": items_obs,
        }
        
        if self.location_memory:
            memorable_location_obs = np.array([0, 0, 0], dtype=np.uint8)
            if len(self.location_memory) > 0:
                loc = self.location_memory.get_prioritized_locations()[0]
                memorable_location_obs = np.array(loc, dtype=np.uint8)
            obs_dict["memorable_location"] = memorable_location_obs
            
        return obs_dict

    def _read_key_event_flags(self):
        flags = []
        for event_name, (addr, bit) in KEY_EVENT_FLAGS_DEF.items():
            byte = self.pyboy.memory[addr]
            flag_value = (byte >> bit) & 1
            flags.append(flag_value)
        return np.array(flags, dtype=np.uint8)

    def _save_memory_screenshot(self, coords):
        image = self.pyboy.screen_image()
        map_id, x, y = coords
        image_path = f'logs/memory_images/map_{map_id}_x_{x}_y_{y}_{self.step_count}.png'
        image.save(image_path)
        return image_path
    
    def _calculate_reward(self, obs, action):
        reward_components = [(TIME_PENALTY, "Time Penalty")]
        self._calculate_dialogue_reward(obs, action, reward_components)
        self._calculate_team_balance_penalty(reward_components)
        if self.lesson == "exploration":
            self._calculate_exploration_reward(obs, action, reward_components)
        elif self.lesson == "battle":
            self._calculate_battle_reward(obs, action, reward_components)
        elif self.lesson == "healer":
            self._calculate_healer_reward(obs, action, reward_components)
        elif self.lesson == "shopping":
            self._calculate_shopping_reward(obs, action, reward_components)
        elif self.lesson == "inventory":
            self._calculate_inventory_reward(obs, action, reward_components)
        elif self.lesson == "switch":
            self._calculate_switch_reward(obs, action, reward_components)
        elif self.lesson == "manager":
            self._calculate_exploration_reward(obs, action, reward_components)
            self._calculate_battle_reward(obs, action, reward_components)
            self._calculate_healer_reward(obs, action, reward_components)
            self._calculate_shopping_reward(obs, action, reward_components)
            self._calculate_inventory_reward(obs, action, reward_components)
            self._calculate_switch_reward(obs, action, reward_components)
            self._calculate_event_reward(obs, action, reward_components)
        else:
            self._calculate_exploration_reward(obs, action, reward_components)
            self._calculate_battle_reward(obs, action, reward_components)
            self._calculate_healer_reward(obs, action, reward_components)
            self._calculate_shopping_reward(obs, action, reward_components)
            self._calculate_inventory_reward(obs, action, reward_components)
            self._calculate_switch_reward(obs, action, reward_components)
            self._calculate_event_reward(obs, action, reward_components)
        # DEBUG: Print all reward components
        # print(f"DEBUG: Reward Components: {reward_components}")
        total_reward = sum(val for val, desc in reward_components)
        self.last_reward_components = reward_components
        return total_reward

    def _calculate_team_balance_penalty(self, reward_components):
        party_levels = self.quest_flags.get("party_levels", [0]*6)
        active_levels = [lvl for lvl in party_levels if lvl > 0]
        if len(active_levels) >= 2: 
            max_level = max(active_levels)
            min_level = min(active_levels)
            level_spread = max_level - min_level
            penalty = -0.01 * level_spread
            if penalty < 0:
                reward_components.append((penalty, "Unbalanced Team"))

    def _calculate_exploration_reward(self, obs, action, reward_components):
        stats = obs["stats"]
        current_map = stats[2]
        script_state = stats[3]
        menu_state = obs["menu_state"]
        current_menu = menu_state[0]
        if self.target_coords:
            target_map, target_x, target_y = self.target_coords
            current_x, current_y = stats[0], stats[1]
            if current_map == target_map:
                last_x, last_y = self.last_coords[1], self.last_coords[2]
                dist_before = abs(last_x - target_x) + abs(last_y - target_y)
                dist_after = abs(current_x - target_x) + abs(current_y - target_y)
                if dist_after < dist_before:
                    reward_components.append((0.2, "Closer to Target"))
                else:
                    reward_components.append((-0.2, "Farther from Target"))
                if dist_after == 0:
                    reward_components.append((10.0, "Reached Target"))
                    if self.location_memory:
                        self.location_memory.decay_location(self.target_coords)
                    self.target_coords = None
            else:
                reward_components.append((-0.5, "Wrong Map for Target"))
        if current_menu != 0 and not self.quest_flags.get("in_battle", False):
            reward_components.append((-0.05, "Menu Open"))
        coords = (current_map, stats[0], stats[1])
        if coords not in self.visited_tiles:
            reward_components.append((0.25, "New Tile")) # Adjusted from 0.10 to 0.25
            self.visited_tiles.add(coords)
        if current_map not in self.visited_maps:
            reward_components.append((1.0, "New Map"))
            self.visited_maps.add(current_map)
            self.visited_tiles.clear()
        self.quest_flags["last_map"] = current_map
        if not self.quest_flags["game_started"] and current_map != self.title_screen_map_id:
            reward_components.append((10.0, "Game Start"))
            self.quest_flags["game_started"] = True
        if (self.quest_flags["game_started"] and script_state == 0 and
            self.stagnation_counter >= STAGNATION_THRESHOLD):
            reward_components.append((STAGNATION_PENALTY, "Stuck"))

    def _calculate_battle_reward(self, obs, action, reward_components):
        stats = obs["stats"]
        opponent_hp = stats[5]
        party_current_hp = self.quest_flags["party_hp"]
        party_max_hp = self.quest_flags["party_max_hp"]
        prev_party_hp = self.quest_flags.get("prev_party_hp", party_current_hp)
        
        # 1. Damage Reward/Penalty
        if opponent_hp < self.quest_flags["opponent_hp"]:
            dmg_dealt = (self.quest_flags["opponent_hp"] - opponent_hp) * 1.0
            reward_components.append((dmg_dealt, "Dmg Dealt"))
        
        for i in range(len(party_current_hp)):
            if party_current_hp[i] < prev_party_hp[i]:
                dmg = (prev_party_hp[i] - party_current_hp[i]) * 0.2
                reward_components.append((-dmg, f"Dmg Taken (Pkmn {i})"))
            if party_current_hp[i] == 0 and prev_party_hp[i] > 0:
                reward_components.append((-10.0, f"Fainted (Pkmn {i})"))
        if opponent_hp == 0 and self.quest_flags["opponent_hp"] > 0:
            reward_components.append((100.0, "Beat Opponent"))
        self.quest_flags["opponent_hp"] = max(opponent_hp, 0)
        self.quest_flags["prev_party_hp"] = party_current_hp
        last_text_hash = self.quest_flags.get("last_text_hash", 0)
        
        # 2. Text Hash Rewards (Optimized Values)
        if last_text_hash == self.empty_dialogue_hash:
            pass 
        elif last_text_hash == HASH_SUPER_EFFECTIVE:
            reward_components.append((10.0, "Super-Effective Hit"))
        elif last_text_hash == HASH_NOT_VERY_EFFECTIVE:
            reward_components.append((-5.0, "Not Effective Hit"))
        elif last_text_hash == HASH_NO_EFFECT:
            reward_components.append((-15.0, "No Effect Hit")) 
        
        self.quest_flags["last_text_hash"] = self.empty_dialogue_hash
        battle_type = self.pyboy.memory[ADDR_BATTLE_TYPE]
        in_battle_flag = self.quest_flags.get("in_battle", False)
        
        # 3. Cowardly Run Logic (Unchanged)
        if in_battle_flag and battle_type == 0: 
            if self.quest_flags["opponent_hp"] > 0: 
                total_current_hp = sum(party_current_hp)
                total_max_hp = sum(party_max_hp)
                hp_percent = 1.0
                if total_max_hp > 0:
                    hp_percent = total_current_hp / total_max_hp
                if hp_percent > 0.6:
                    reward_components.append((-5.0, "Cowardly Run")) 
                elif hp_percent > 0.5:
                    reward_components.append((1.0, "Run (Low HP)"))
                elif hp_percent > 0.4:
                    reward_components.append((2.0, "Run (Mid HP)"))
                else:
                    reward_components.append((4.0, "Run (Critical HP)"))
        self.quest_flags["in_battle"] = (battle_type > 0)
        
    def _calculate_healer_reward(self, obs, action, reward_components):
        party_hp = self.quest_flags["party_hp"]
        party_count = self.quest_flags["party_count"]
        hp_gained = False
        for i in range(party_count):
            if party_hp[i] > self.quest_flags["party_hp"][i]:
                hp_gained = True
                break
        nurse_joy_script_val = self.pyboy.memory[ADDR_NURSE_JOY_SCRIPT]
        if (hp_gained and nurse_joy_script_val > 0):
            reward_components.append((10.0, "Healed @ PC"))
            if self.location_memory:
                stats = obs["stats"]
                coords = (stats[2], stats[0], stats[1])
                image_path = self._save_memory_screenshot(coords)
                self.location_memory.add_location(coords, image_path=image_path)

    def _calculate_shopping_reward(self, obs, action, reward_components):
        items_obs = obs["items"]
        current_items = {}
        for i in range(ITEM_BAG_SIZE):
            item_id = items_obs[i*2]
            item_count = items_obs[i*2 + 1]
            if item_id != 0:
                current_items[item_id] = item_count
        if not hasattr(self, 'last_item_counts'):
            self.last_item_counts = current_items
        for item_id, count in current_items.items():
            if item_id not in self.last_item_counts:
                 reward_components.append((2.0, "Got New Item Type"))
                 if self.location_memory:
                    stats = obs["stats"]
                    coords = (stats[2], stats[0], stats[1])
                    image_path = self._save_memory_screenshot(coords)
                    self.location_memory.add_location(coords, image_path=image_path)
            elif count > self.last_item_counts.get(item_id, 0):
                 reward_components.append((0.5, "Got More Items"))
        self.last_item_counts = current_items

    def _calculate_inventory_reward(self, obs, action, reward_components):
        party_hp = self.quest_flags["party_hp"]
        party_count = self.quest_flags["party_count"]
        hp_gained = False
        for i in range(party_count):
            if party_hp[i] > self.quest_flags["party_hp"][i]:
                hp_gained = True
                break
        if hp_gained:
            reward_components.append((5.0, "Used Potion"))
            if self.location_memory:
                stats = obs["stats"]
                coords = (stats[2], stats[0], stats[1])
                image_path = self._save_memory_screenshot(coords)
                self.location_memory.add_location(coords, image_path=image_path)
        current_party_status = self.quest_flags["party_status"]
        prev_party_status = self.quest_flags.get("prev_party_status", current_party_status)
        prev_status_active = prev_party_status[0]
        curr_status_active = current_party_status[0]
        if (prev_status_active & STATUS_POISON) != 0 and (curr_status_active & STATUS_POISON) == 0:
            reward_components.append((10.0, "Used Antidote"))
        if (prev_status_active & STATUS_PARALYSIS) != 0 and (curr_status_active & STATUS_PARALYSIS) == 0:
            reward_components.append((10.0, "Used Parlyz Heal"))
            
    def _calculate_switch_reward(self, obs, action, reward_components):
        current_species = self.quest_flags["party_species"]
        if not hasattr(self, 'last_party_species'):
            self.last_party_species = current_species
        if current_species[0] != self.last_party_species[0] and self.last_party_species[0] != 0:
            reward_components.append((3.0, "Switched Pokemon"))
        self.last_party_species = current_species

    def _calculate_dialogue_reward(self, obs, action, reward_components):
        script_state = obs["stats"][3] 
        if script_state > 0 and action == 4: # Action 4 is 'A'
            text_bytes = self.pyboy.memory[ADDR_TEXT_BUFFER_START : ADDR_TEXT_BUFFER_END]
            text_hash = hash(tuple(text_bytes))
            if text_hash not in self.seen_dialogue_hashes and text_hash != self.empty_dialogue_hash:
                self.seen_dialogue_hashes.add(text_hash)
                reward_components.append((0.5, "New Dialogue"))

    def _calculate_event_reward(self, obs, action, reward_components):
        stats = obs["stats"]
        badges = stats[4]
        key_events = obs["key_events"]
        party_count = self.pyboy.memory[0xD163]
        party_levels = self.quest_flags["party_levels"]
        party_exp = self.quest_flags["party_exp"]
        pokedex_caught = sum(bin(b).count('1') for b in self.pyboy.memory[ADDR_POKEDEX_CAUGHT:ADDR_POKEDEX_CAUGHT + 19])
        money_bytes = self.pyboy.memory[ADDR_MONEY_HI : ADDR_MONEY_LO + 1]
        money = self._bcd_to_int(money_bytes)
        
        # --- KEY EVENT REWARD ADJUSTMENTS ---
        parcel_idx = list(KEY_EVENT_FLAGS_DEF.keys()).index("GOT_OAKS_PARCEL")
        if key_events[parcel_idx] == 1 and self.quest_flags["key_events"][parcel_idx] == 0:
            reward_components.append((25.0, "Delivered Parcel")) # Adjusted from 10.0
        brock_idx = list(KEY_EVENT_FLAGS_DEF.keys()).index("BEAT_BROCK")
        if key_events[brock_idx] == 1 and self.quest_flags["key_events"][brock_idx] == 0:
            reward_components.append((250.0, "Beat Brock")) # Adjusted from 100.0
        
        if party_count > self.quest_flags.get("party_count", 0) and self.quest_flags.get("party_count", 0) > 0:
             reward_components.append((2.0, "Got PKMN"))
        elif party_count > self.quest_flags.get("party_count", 0) and self.quest_flags.get("party_count", 0) == 0:
            reward_components.append((10.0, "Got 1st PKMN"))
        for i in range(party_count):
            if party_levels[i] > self.quest_flags.get("party_levels", [0]*6)[i]:
                lvl_reward = (party_levels[i] - self.quest_flags.get("party_levels", [0]*6)[i]) * 1.5
                reward_components.append((lvl_reward, f"Level Up (Pkmn {i})"))
            if party_exp[i] > self.quest_flags.get("party_exp", [0]*6)[i]:
                exp_reward = (party_exp[i] - self.quest_flags.get("party_exp", [0]*6)[i]) * 0.005
                if exp_reward > 0.0001:
                    reward_components.append((exp_reward, f"EXP Gain (Pkmn {i})"))
        if pokedex_caught > self.quest_flags.get("pokedex_caught", 0):
            reward_components.append((5.0, "Caught PKMN"))
        if money > self.quest_flags.get("money", 0):
            money_reward = (money - self.quest_flags.get("money", 0)) * 0.0001
            if money_reward > 0:
                reward_components.append((money_reward, "Got Money"))
        new_events = np.sum(key_events - self.quest_flags.get("key_events", 0) > 0)
        if new_events > 0:
            reward_components.append((new_events * 25.0, "Key Event")) # Adjusted generic multiplier from 5.0 to 25.0
        self.quest_flags.update({
            "party_count": party_count,
            "pokedex_caught": pokedex_caught,
            "badges": badges,
            "money": money,
            "key_events": key_events,
        })

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        pre_step_obs = self._get_observation()
        self.quest_flags["prev_party_status"] = self.quest_flags["party_status"]
        pre_step_active_status = self.quest_flags["prev_party_status"][0]
        pre_step_in_battle = self.quest_flags.get("in_battle", False)
        pre_step_items = pre_step_obs["items"]
        current_stats = pre_step_obs["stats"]
        current_menu = pre_step_obs["menu_state"][0]
        current_cursor = pre_step_obs["menu_state"][1]
        script_state = current_stats[3]
        if current_menu != 0:
            if current_cursor == self.quest_flags["cursor_pos"]:
                 self.menu_stagnation_counter += 1
            else:
                 self.menu_stagnation_counter = 0
        elif script_state != 0:
            if action == 4 or action == 5: # A or B
                self.menu_stagnation_counter = 0
            else:
                self.menu_stagnation_counter += 1
        else:
            self.menu_stagnation_counter = 0
        if current_menu != 0 or current_stats[3] != 0:
            self.stagnation_counter = 0
        else:
            coords = (current_stats[2], current_stats[0], current_stats[1])
            if coords == self.last_coords:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_coords = coords
        prev_x = pre_step_obs["stats"][0]
        prev_y = pre_step_obs["stats"][1]
        frames_to_hold = 3
        frames_to_wait = 6 if self.headless else 25
        press_event = ACTION_PRESS_MAP.get(action)
        release_event = ACTION_RELEASE_MAP.get(action)
        if not self.headless:
            if press_event:
                self.pyboy.send_input(press_event)
                for _ in range(frames_to_hold):
                    self.pyboy.tick()
                self.pyboy.send_input(release_event)
                for _ in range(frames_to_wait):
                    self.pyboy.tick()
            else:
                for _ in range(frames_to_hold + frames_to_wait):
                    self.pyboy.tick()
        else:
            if press_event:
                self.pyboy.send_input(press_event)
                self.pyboy.tick(frames_to_hold)
                self.pyboy.send_input(release_event)
                self.pyboy.tick(frames_to_wait)
            else:
                self.pyboy.tick(frames_to_hold + frames_to_wait)
        text_bytes = self.pyboy.memory[ADDR_TEXT_BUFFER_START : ADDR_TEXT_BUFFER_END]
        self.quest_flags["last_text_hash"] = hash(tuple(text_bytes))
        obs = self._get_observation()
        new_x = obs["stats"][0]
        new_y = obs["stats"][1]
        status_penalty = 0.0
        status_penalty_reason = ""
        if pre_step_in_battle:
            item_ids = pre_step_items[::2] 
            has_antidote = (ITEM_ID_ANTIDOTE in item_ids)
            has_parlyz_heal = (ITEM_ID_PARLYZ_HEAL in item_ids)
            if (pre_step_active_status & STATUS_POISON) != 0 and has_antidote:
                status_penalty = -0.1
                status_penalty_reason = "Not Using Antidote"
            elif (pre_step_active_status & STATUS_PARALYSIS) != 0 and has_parlyz_heal:
                status_penalty = -0.1
                status_penalty_reason = "Not Using Parlyz Heal"
        reward = self._calculate_reward(obs, action)
        if action in [0, 1, 2, 3]: # UP, DOWN, LEFT, RIGHT
            # (NEW) Added debug print
            print(f"DEBUG: Action={action}, Prev=({prev_x},{prev_y}), New=({new_x},{new_y}), InMenu={current_menu}, ScriptState={script_state}")
            if new_x == prev_x and new_y == prev_y and script_state == 0 and current_menu == 0:
                reward -= 0.05 # Apply bump penalty
                self.last_reward_components.append((-0.05, "Bumped Wall"))
        
        # Ensure status penalty is correctly added to the total reward and logged
        reward += status_penalty
        if status_penalty < 0:
            self.last_reward_components.append((status_penalty, status_penalty_reason))
            
        terminated = False
        truncated = self.step_count >= self.max_steps
        if self.lesson == "battle":
            is_in_battle = self.quest_flags.get("in_battle", False)
            if pre_step_in_battle and not is_in_battle:
                terminated = True
        if truncated:
            self.visited_tiles.clear()
            self.stagnation_counter = 0
        info = {
            "menu_stagnation_counter": self.menu_stagnation_counter,
            "stagnation_counter": self.stagnation_counter,
            "reward_triggers": self.last_reward_components, # This list contains ALL calculated rewards
            "reward_trigger": "None",
            "map_id": obs["stats"][2],
            "x": obs["stats"][0],
            "y": obs["stats"][1],
        }
        if self.last_reward_components:
            # Find the largest positive reward trigger for debug display convenience
            main_trigger = max(self.last_reward_components, key=lambda item: item[0], default=(0, "None"))
            if main_trigger[0] > 0:
                info["reward_trigger"] = main_trigger[1]
        self.quest_flags.update({
            "opponent_hp": max(obs["stats"][5], 0),
            "opponent_status": self.pyboy.memory[ADDR_OPPONENT_STATUS],
            "last_map": obs["stats"][2],
            "current_menu": obs["menu_state"][0],
            "cursor_pos": obs["menu_state"][1],
            "main_event_flags_1": self.pyboy.memory[ADDR_MAIN_EVENT_FLAGS_1],
        })
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.visited_tiles.clear()
        self.visited_maps.clear()
        self.frame_stack.clear()
        self.exploration_map = np.zeros((EXPLORATION_MAP_SIZE, EXPLORATION_MAP_SIZE), dtype=np.uint8)
        self.seen_dialogue_hashes.clear()
        if not hasattr(self, 'last_item_counts'):
            self.last_item_counts = {}
        if not hasattr(self, 'last_party_species'):
            self.last_party_species = [0]*6
        if not self.state_paths:
             raise FileNotFoundError(f"No state files found for lesson '{self.lesson}'. Searched for: {os.path.join(STATE_DIR, self.lesson + '_*.state')}")
        selected_state_path = random.choice(self.state_paths)
        if not os.path.exists(selected_state_path):
             raise FileNotFoundError(f"Selected state file {selected_state_path} not found for lesson '{self.lesson}'.")
        if not self.headless:
             print(f"--- Loading state: {selected_state_path} ---")
        with open(selected_state_path, "rb") as f:
            self.pyboy.load_state(f)
        for release_event in ACTION_RELEASE_MAP.values():
            if release_event:
                self.pyboy.send_input(release_event)
        self.pyboy.tick(10)
        obs = self._get_observation()
        self.title_screen_map_id = obs["stats"][2]
        initial_opponent_hp = obs["stats"][5]
        initial_battle_type = self.pyboy.memory[ADDR_BATTLE_TYPE]
        money_bytes = self.pyboy.memory[ADDR_MONEY_HI : ADDR_MONEY_LO + 1]
        initial_money = self._bcd_to_int(money_bytes)
        self.quest_flags.update({
            "game_started": (self.lesson not in ["exploration", "manager"]), 
            "party_count": self.pyboy.memory[0xD163],
            "pokedex_caught": sum(bin(b).count('1') for b in self.pyboy.memory[ADDR_POKEDEX_CAUGHT:ADDR_POKEDEX_CAUGHT + 19]),
            "badges": obs["stats"][4],
            "opponent_hp": initial_opponent_hp,
            "opponent_status": self.pyboy.memory[ADDR_OPPONENT_STATUS],
            "last_map": self.title_screen_map_id,
            "money": initial_money,
            "key_events": obs["key_events"],
            "current_menu": obs["menu_state"][0],
            "cursor_pos": obs["menu_state"][1],
            "main_event_flags_1": self.pyboy.memory[ADDR_MAIN_EVENT_FLAGS_1],
            "in_battle": (initial_battle_type > 0),
            "party_max_hp": self.quest_flags.get("party_max_hp", [0]*6),
            "prev_party_hp": self.quest_flags.get("party_hp", [0]*6),
            "party_status": self.quest_flags.get("party_status", [0]*6),
            "prev_party_status": self.quest_flags.get("party_status", [0]*6),
            "last_text_hash": 0,
        })
        self.stagnation_counter = 0
        self.menu_stagnation_counter = 0
        self.last_coords = (obs["stats"][2], obs["stats"][0], obs["stats"][1])
        self.visited_tiles.add(self.last_coords)
        info = {
            "menu_stagnation_counter": self.menu_stagnation_counter,
            "stagnation_counter": self.stagnation_counter,
            "reward_triggers": [],
            "reward_trigger": "None",
            "map_id": obs["stats"][2],
            "x": obs["stats"][0],
            "y": obs["stats"][1],
        }
        return obs, info

    def close(self):
        if self.location_memory:
            self.location_memory.save_memory()
        self.pyboy.stop()
        print("PyBoy environment closed.")
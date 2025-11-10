# src/env/wrappers.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2 # Import OpenCV for the debug window
import json
import os
import threading # Import threading
import tkinter as tk # Import tkinter
from src.utils.gui_memory_scanner import GUIMemoryScanner # Import scanner

class TransposeObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper to transpose the 'vision' observation from (Height, Width, Channel) 
    to (Channel, Height, Width).
    """
    def __init__(self, env):
        super().__init__(env)
        
        vision_space = self.observation_space.spaces["vision"]
        h, w, c = vision_space.shape
        new_vision_shape = (c, h, w)
        
        new_vision_space = spaces.Box(
            low=0,
            high=255,
            shape=new_vision_shape,
            dtype=np.uint8
        )
        
        new_spaces = self.observation_space.spaces.copy()
        new_spaces["vision"] = new_vision_space
        
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        obs["vision"] = np.transpose(obs["vision"], (2, 0, 1))
        return obs

class InfoInjectorWrapper(gym.Wrapper):
    """
    Injects keys from the obs dict and info dict into the top-level info.
    """
    def __init__(self, env, keys_to_inject=("menu_state", "menu_stagnation_counter", 
                                            "stagnation_counter", "reward_trigger", 
                                            "reward_triggers"), location_memory=None):
        super().__init__(env)
        self.keys_to_inject = keys_to_inject
        self.location_memory = location_memory
        self._obs_keys = []
        self._info_keys = []

        base_obs_space = env.observation_space
        if hasattr(env, "unwrapped"):
             base_obs_space = env.unwrapped.observation_space

        if isinstance(base_obs_space, gym.spaces.Dict):
            for k in self.keys_to_inject:
                if k in base_obs_space.spaces:
                    self._obs_keys.append(k)
                else:
                    self._info_keys.append(k)
        else:
             self._info_keys = list(self.keys_to_inject)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self._obs_keys:
            if isinstance(obs, dict) and k in obs:
                val = obs[k]
                try:
                    info[k] = np.array(val, copy=True)
                except Exception:
                    info[k] = val
        
        if self.location_memory is not None:
            info["location_memory"] = self.location_memory

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for k in self._obs_keys:
            if isinstance(obs, dict) and k in obs:
                try:
                    info[k] = np.array(obs[k], copy=True)
                except Exception:
                    info[k] = obs[k]
        
        for k in self._info_keys:
            if k not in info:
                info[k] = 0 if "counter" in k else ("None" if "trigger" in k else [])

        if self.location_memory is not None:
            info["location_memory"] = self.location_memory

        return obs, info

class DebugDashboardWrapper(gym.Wrapper):
    """
    (UPGRADED) Creates an OpenCV window for debug info AND
    launches the Tkinter GUI Memory Scanner in a separate thread.
    """
    def __init__(self, env):
        super().__init__(env)
        self.window_name = "Pokemon AI Debug Dashboard"
        self.width = 400
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.line_height = 25
        
        self.key_event_names = [
            "Got Pokedex", "Got Oak's Parcel", "Beat Brock",
            "Beat Rival (Cerulean)", "Got Bike Voucher", "Got SS Ticket",
        ]
        
        self.STATUS_POISON = 0x08
        self.STATUS_PARALYSIS = 0x40
        
        self.memory_watch_list = []
        self._load_memory_config()
        
        # Calculate height based on all sections
        base_height_lines = 21 
        key_event_lines = len(self.key_event_names) + 2
        
        # --- (FIXED) Increased max lines for reward display ---
        self.MAX_REWARD_DISPLAY_LINES = 20
        reward_lines = self.MAX_REWARD_DISPLAY_LINES + 2
        # --- END FIX ---
        
        memory_lines = len(self.memory_watch_list) + 2
        total_lines = base_height_lines + key_event_lines + reward_lines + memory_lines
        self.height = total_lines * self.line_height
        if self.height < 800: self.height = 800

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        self.is_paused = False
        self.is_human_control = False
        
        initial_speed = 1
        if hasattr(self.unwrapped, 'pyboy'):
             self.unwrapped.pyboy.set_emulation_speed(initial_speed)
             
        cv2.createTrackbar('Speed (1-10)', self.window_name, initial_speed, 10, self._on_speed_change)
        cv2.createTrackbar('Pause (0=No, 1=Yes)', self.window_name, 0, 1, self._on_pause_change)
        cv2.createTrackbar('Human Ctrl (0=AI, 1=You)', self.window_name, 0, 1, self._on_human_control_change)
        
        self.last_obs = None
        self.last_info = {}
        
        # Launch GUI Scanner (DISABLED to prevent thread conflicts in watcher)
        # self.scanner_thread = threading.Thread(target=self._launch_scanner_gui, daemon=True)
        # self.scanner_thread.start()

    def _launch_scanner_gui(self):
        try:
            print("Launching GUI Memory Scanner thread...")
            root = tk.Tk()
            # This requires a PyBoy instance to be present in the unwrapped env
            pyboy_ref = self.unwrapped.pyboy if hasattr(self.unwrapped, 'pyboy') else None
            if pyboy_ref:
                scanner_app = GUIMemoryScanner(root, pyboy_ref)
                root.protocol("WM_DELETE_WINDOW", scanner_app.on_closing)
                root.mainloop()
            else:
                 print("Error: PyBoy instance not found in unwrapped env for scanner.")
            print("GUI Memory Scanner thread finished.")
        except Exception as e:
            print(f"Error launching GUI Memory Scanner: {e}")
            print("If on Linux, you may need 'sudo apt-get install python3-tk'")


    def _load_memory_config(self):
        config_path = "debug_memory_config.json"
        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found. Memory watcher will be empty.")
            return
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                watch_list = config.get("watch_list", [])
                for item in watch_list:
                    if "name" in item and "address" in item:
                        try:
                            address_int = int(item["address"], 16)
                            self.memory_watch_list.append((item["name"], address_int))
                        except ValueError:
                            print(f"Error: Invalid address format '{item['address']}' in config.")
        except Exception as e:
            print(f"Error loading {config_path}: {e}")

    # (NEW) Helper to read trackbars without blocking
    def _check_trackbars(self):
        """ Reads all trackbar values and updates states. """
        try:
            self._on_speed_change(cv2.getTrackbarPos('Speed (1-10)', self.window_name))
            self._on_pause_change(cv2.getTrackbarPos('Pause (0=No, 1=Yes)', self.window_name))
            self._on_human_control_change(cv2.getTrackbarPos('Human Ctrl (0=AI, 1=You)', self.window_name))
        except cv2.error:
            # This can happen if the window is closed
            pass

    def _on_speed_change(self, speed_val):
        if speed_val < 1: speed_val = 1
        if hasattr(self.unwrapped, 'pyboy'):
            self.unwrapped.pyboy.set_emulation_speed(speed_val)

    def _on_pause_change(self, val):
        # (NEW) Only print when the state *changes*
        new_state = (val == 1)
        if new_state != self.is_paused:
            self.is_paused = new_state
            if self.is_paused:
                print("\n--- SIMULATION PAUSED (set trackbar to 0 to resume) ---")
            else:
                print("\n--- SIMULATION RESUMED ---")

    def _on_human_control_change(self, val):
        # (NEW) Only print when the state *changes*
        new_state = (val == 1)
        if new_state != self.is_human_control:
            self.is_human_control = new_state
            if self.is_human_control:
                print("\n--- HUMAN CONTROL ENGAGED (Use game window keys: Arrows, Z, X) ---")
            else:
                print("\n--- AI CONTROL ENGAGED ---")

    def step(self, action):
        """
        (UPGRADED) This step function now handles pause and human control.
        """
        # 1. Draw the dashboard based on the *previous* step's obs/info
        self._update_dashboard(self.last_obs, self.last_info) 

        # 2. Handle Pause Loop (blocks AI and Game)
        while self.is_paused:
            # Keep the dashboard window responsive
            if cv2.waitKey(50) & 0xFF == ord('q'):
                self.is_paused = False
                cv2.setTrackbarPos('Pause (0=No, 1=Yes)', self.window_name, 0)
                break
            # Allow user to toggle other controls while paused
            self._check_trackbars()
        
        # 3. Determine which action to execute
        if self.is_human_control:
            # Human is in control.
            # Discard the AI's action, send "No-Op" (7).
            # The underlying env.step() will call pyboy.tick(),
            # allowing the SDL2 window to process human input.
            action_to_execute = 7 
        else:
            # AI is in control.
            action_to_execute = action

        # 4. Execute the step
        obs, reward, terminated, truncated, info = self.env.step(action_to_execute)

        # 5. Store results for the *next* frame's dashboard
        self.last_obs = obs
        self.last_info = info
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        self._update_dashboard(obs, info)
        return obs, info

    def _get_status_string(self, status_byte):
        if status_byte == 0: return "OK"
        s = []
        if (status_byte & self.STATUS_POISON): s.append("PSN")
        if (status_byte & self.STATUS_PARALYSIS): s.append("PAR")
        if not s: return f"OTHER ({status_byte})"
        return ", ".join(s)

    def _update_dashboard(self, obs, info):
        """
        (UPGRADED) This function ONLY draws the dashboard and checks trackbars.
        It does NOT block execution.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        line_type = 2
        white, green, red, yellow, cyan = (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)
        y = 30
        dy = self.line_height
        
        def draw_text(text, color=white, y_pos=None):
            nonlocal y
            if y_pos is not None:
                cv2.putText(frame, text, (10, y_pos), self.font, self.font_scale, color, line_type)
            else:
                cv2.putText(frame, text, (10, y), self.font, self.font_scale, color, line_type)
                y += dy
            
        def draw_header(text):
            nonlocal y
            y += 5 
            draw_text(text, color=yellow)
            y += 5

        if obs is None:
            draw_text("Waiting for reset...")
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        # --- Controls ---
        draw_header("--- CONTROLS ---")
        draw_text(f"Speed: {cv2.getTrackbarPos('Speed (1-10)', self.window_name)}")
        draw_text(f"Pause: {'YES' if self.is_paused else 'NO'}")
        draw_text(f"Human Ctrl: {'ACTIVE' if self.is_human_control else 'AI'}")
        
        # --- General Info ---
        draw_header("--- GENERAL ---")
        draw_text(f"Map ID: {obs['stats'][2]}")
        draw_text(f"Coords: (X={obs['stats'][0]}, Y={obs['stats'][1]})")
        in_battle = "Yes" if self.unwrapped.quest_flags.get("in_battle", False) else "No"
        draw_text(f"In Battle: {in_battle}")
        money = self.unwrapped.quest_flags.get("money", 0)
        draw_text(f"Money: ${money}")
        draw_text(f"Stagnation: {info.get('stagnation_counter', 0)}")
        draw_text(f"Total Reward: {sum(val for val, name in info.get('reward_triggers', [])):.2f}", color=cyan)
        
        # --- Party Info ---
        draw_header("--- PARTY ---")
        party_hp = self.unwrapped.quest_flags.get("party_hp", [])
        party_max_hp = self.unwrapped.quest_flags.get("party_max_hp", [])
        party_status = self.unwrapped.quest_flags.get("party_status", [])
        party_levels = self.unwrapped.quest_flags.get("party_levels", [])
        for i in range(len(party_hp)):
            if i >= len(party_max_hp) or i >= len(party_status) or i >= len(party_levels): break 
            if party_max_hp[i] > 0:
                hp_str = f"HP: {party_hp[i]} / {party_max_hp[i]}"
                lvl_str = f"Lvl: {party_levels[i]}"
                status_str = self._get_status_string(party_status[i])
                color = green if status_str == "OK" else red
                draw_text(f"P{i+1}: {lvl_str} | {hp_str}")
                draw_text(f"     Status: {status_str}", color=color)
        
        # --- Key Events ---
        draw_header("--- KEY EVENTS ---")
        key_events = obs["key_events"]
        for i in range(len(key_events)):
            event_name = self.key_event_names[i] if i < len(self.key_event_names) else f"Event {i}"
            status = "COMPLETED" if key_events[i] == 1 else "---"
            color = green if key_events[i] == 1 else white
            draw_text(f"{event_name}: {status}", color=color)
            
        # --- Last Rewards (FIXED: Full logging) ---
        draw_header(f"--- LAST REWARDS ({len(info.get('reward_triggers', []))} items) ---")
        rewards = info.get("reward_triggers", [])
        
        # Sort by value (descending) for readability if over 20 entries
        rewards_to_display = sorted(rewards, key=lambda x: x[0], reverse=True)
        rewards_to_display = rewards_to_display[:self.MAX_REWARD_DISPLAY_LINES]
        
        if not rewards_to_display:
            draw_text("None")
        else:
            for val, name in rewards_to_display:
                if val > 0:
                    color = green
                elif val < 0:
                    color = red
                else:
                    color = white
                draw_text(f"{val:+.2f} : {name}", color=color)
        
        # --- Dynamic Memory Watch (from config) ---
        draw_header("--- MEMORY WATCH ---")
        if not self.memory_watch_list:
            draw_text("No addresses in config file.")
        else:
            pyboy_mem = self.unwrapped.pyboy.memory
            for name, address in self.memory_watch_list:
                try:
                    value = pyboy_mem[address]
                    val_str = f"0x{value:02X} ({value})"
                    draw_text(f"{name:<18}: {val_str}")
                except Exception:
                    draw_text(f"{name:<18}: ERROR")
        
        # (NEW) Display "HUMAN CONTROL" message if active
        if self.is_human_control:
            alert_y = 65 # y-position for the alert
            draw_text("!!! HUMAN CONTROL ACTIVE !!!", color=red, y_pos=alert_y)
            draw_text(" (Use Game Window Keys)", color=white, y_pos=alert_y + dy)

        # --- Display and Non-Blocking Wait ---
        cv2.imshow(self.window_name, frame)
        
        # (NEW) Check trackbars, but don't block
        self._check_trackbars() 
        cv2.waitKey(1) # Must have a small wait to keep window responsive

    def close(self):
        self.env.close()
        try:
            cv2.destroyWindow(self.window_name)
        except Exception as e:
            print(f"Error closing debug window: {e}")
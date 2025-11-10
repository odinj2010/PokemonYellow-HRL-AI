# debug_text_hash.py
#
# This is a utility script to find the memory hashes of in-game text.
# Use this to find the hashes for "It's super-effective!", etc.

import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import os
import keyboard # You may need to run: pip install keyboard

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
# (CHANGE THIS) Load a battle state file to start
STATE_PATH = os.path.join(STATE_DIR, "battle_08.state")

ADDR_TEXT_BUFFER_START = 0xC5B0
ADDR_TEXT_BUFFER_END = 0xC6B0 # 256 bytes long

# --- How to Use ---
# 1. Run this script: python debug_text_hash.py
# 2. The game will load. Play the game (press 'A', etc.)
# 3. When the text you want to capture is on screen
#    (e.g., "It's super-effective!"),
#    PRESS THE 'H' KEY in your console.
# 4. The script will print the memory hash for that text.
# 5. Copy that hash and paste it into the constants in pokemon_env.py

def main():
    if not os.path.exists(STATE_PATH):
        print(f"Error: Save state not found at {STATE_PATH}")
        print(f"Please create 'battle_01.state' or change the STATE_PATH variable.")
        return

    print("--- Text Hash Debugger ---")
    print(f"Loading state: {STATE_PATH}")
    print(f"Controls: Z=A, X=B, Enter=Start, Arrows=Move")
    print(f"\n>>> PRESS 'H' IN THIS CONSOLE <<<")
    print(f"    when text is on screen to capture its hash.")
    print(f"Press Ctrl+C to quit.")

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)
    
    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(f)
        
    def get_hash():
        text_bytes_list = pyboy.memory[ADDR_TEXT_BUFFER_START : ADDR_TEXT_BUFFER_END]
        
        # (FIXED) Convert the list of integers into a bytes object
        text_bytes = bytes(text_bytes_list)
        
        text_hash = hash(tuple(text_bytes_list))
        
        # Now we can call .hex() on the bytes object
        text_hex = text_bytes.hex()
        
        print("\n---------------------------------")
        print(f"Captured Text Hash: {text_hash}")
        print(f"Captured Hex: {text_hex}")
        print("---------------------------------")

    # Register the hotkey
    keyboard.add_hotkey('h', get_hash)

    try:
        while True:
            pyboy.tick()
            
    except KeyboardInterrupt:
        print("\nDebugger stopped.")
    finally:
        pyboy.stop()

if __name__ == "__main__":
    main()
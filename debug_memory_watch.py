# debug_memory_watch.py
#
# This is a utility script to find and watch RAM addresses.
# Use this to find the real X and Y coordinates.
#-------------------#
# --- HOW TO USE ---#
#-------------------#
# 1. Run this script: python debug_memory_watch.py
# 2. The game will load. Play the game (press 'A', etc.)
# 3. Watch the console for real-time memory address values.
# 4. Press Ctrl+C to quit.

# --- Imports ---

import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent 
import os
import keyboard 

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
# Load your new save state
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")

# --- Addresses to Watch ---

ADDRESS_WATCH_LIST = {
    "Map ID (Correct)": 0xD35D,
    "Player Y (Map)": 0xD360,
    "Player X (Map)": 0xD361,
    "test 1": 0xF35D,
    "test 2": 0xF360,
    "test 3": 0xF361,
    "Script State": 0xC506,
    "Battle Type": 0xD05A,
    "Current Menu": 0xD057,
    "Money 1": 0xD347,
    "Money 2": 0xD348,
    "Money 3": 0xD349,
}

# --- Main Function ---

def main():
    if not os.path.exists(STATE_PATH):
        print(f"Error: Save state not found at {STATE_PATH}")
        print("Please create 'new_game.state' using create_save_state.py")
        return

    print("--- Real-Time Memory Watcher ---")
    print(f"Loading state: {STATE_PATH}")
    print(f"Controls: Z=A, X=B, Enter=Start, Arrows=Move")
    print(f"\n>>> Watching {len(ADDRESS_WATCH_LIST)} addresses... <<<")
    print("Play the game and watch the values change in this console.")
    print(f"Press Ctrl+C to quit.")

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1) # Run at normal speed
    
    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(f)
        
    try:
        header = " | ".join([f"{name:^18}" for name in ADDRESS_WATCH_LIST.keys()])
        print(header)
        print("-" * len(header))
        
        last_values = {}
        
        while True:
            pyboy.tick()
            
            current_values = {}
            changed = False
            
            # Read all memory values
            for name, addr in ADDRESS_WATCH_LIST.items():
                val = pyboy.memory[addr]
                current_values[name] = val
                if last_values.get(name) != val:
                    changed = True
            
            # Only print if a value changed to avoid spam
            if changed:
                line_values = []
                for name in ADDRESS_WATCH_LIST.keys():
                    val = current_values[name]
                    # Format as hex (0x00) and decimal (0)
                    val_str = f"0x{val:02X} ({val})"
                    line_values.append(f"{val_str:^18}")
                
                print("\r" + " | ".join(line_values), end="")
            
            last_values = current_values
            time.sleep(0.01) # Small delay to make it readable
            
    except KeyboardInterrupt:
        print("\nDebugger stopped.")
    finally:
        pyboy.stop()

if __name__ == "__main__":
    main()
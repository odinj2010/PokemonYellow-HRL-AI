# create_save_state.py
# (FIXED: Updated to be a general-purpose script for
#  manual play and creating ALL training save states)

import time
from pyboy import PyBoy
import os

# Define the path to the Pokemon Yellow ROM file
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
DEFAULT_STATE_NAME = "Pokemon_Yellow_Version_Special_Pikachu_Edition.state"

# Ensure the 'states' directory exists
os.makedirs(STATE_DIR, exist_ok=True)

# Print instructions for the user
print(f"Loading ROM: {ROM_PATH}")
print("\n--- Manual Play & Save State Creator ---")
print("The game window will open. You can play the game normally.")
print("\n--- CONTROLS ---")
print(f"* **Save State:** Press F2 (or Fn+F2)")
print(f"* **Load State:** Press F1 (or Fn+F1) to load your last save.")
print(f"* **Speed:** Press Backspace to toggle between 1x and 3x speed.")
print(f"* **Movement:** Arrow Keys")
print(f"* **A Button:** Z")
print(f"* **B Button:** X")
print(f"* **Start:** Enter")
print(f"* **Select:** Right Shift")
print(f"\n--- HOW TO CREATE TRAINING DATA ---")
print(f"1. Play the game to a point you want to train (e.g., just before a battle).")
print(f"2. Press F2 to save. This creates a file named '{DEFAULT_STATE_NAME}'")
print(f"   in the main project folder.")
print(f"3. Go to the project folder. Find this file.")
print(f"4. **COPY and RENAME** this file into the '{STATE_DIR}/' folder.")
print(f"\n--- NAMING CONVENTION (CRITICAL!) ---")
print(f"* **Start of Game:** new_game.state")
print(f"* **Battle Lesson:** battle_01.state, battle_02.state, etc.")
print(f"* **Healer Lesson:** healer_01.state, healer_02.state, etc.")
print(f"* **Shopping Lesson:** shopping_01.state, shopping_02.state, etc.")
print(f"* **Inventory Lesson:** inventory_01.state, inventory_02.state, etc.")
print(f"* **Switch Lesson:** switch_01.state, switch_02.state, etc.")
print(f"\nPress Ctrl+C in this console window to quit when finished.")

# Initialize PyBoy instance variable
pyboy = None
try:
    # Create a PyBoy instance with the specified ROM.
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    # Set the emulation speed to normal (1x) so the user can interact.
    pyboy.set_emulation_speed(1)
    
    # Main loop to keep the emulator running
    while True: 
        pyboy.tick()
        
# Handle keyboard interrupt (Ctrl+C) to exit the script
except KeyboardInterrupt:
    print("\nState creation script exiting.")
finally:
    if pyboy:
        pyboy.stop()
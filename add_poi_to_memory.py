# add_poi_to_memory.py
#
# This is a one-time utility script to manually add
# key Points of Interest (POIs) to the AI's location memory.
# This helps the Manager AI learn to navigate between major hubs.

import os
from src.utils.location_memory import LocationMemory

# --- CONFIGURATION ---
# 1. Find the coordinates by playing the game (using watch_specialist.py)
# 2. Update the (MAP_ID, X, Y) tuples below.
#
# (MAP_ID, X, Y)
POINTS_OF_INTEREST = {
    #"viridian_forest_entry": (2, 24, 25), # Example: (Map 2, X 24, Y 25)
    #"viridian_forest_exit": (2, 25, 4),   # Example: (Map 2, X 25, Y 4)
    #"mt_moon_entry": (3, 5, 6),           # Example: (Map 3, X 5, Y 6)
    #"mt_moon_exit": (3, 20, 10),          # Example: (Map 3, X 20, Y 10)
    # Add more for Cerulean City, etc.
}

MEMORY_FILE = "logs/location_memory.pkl"
IMAGE_DIR = "logs/memory_images"

def main():
    print(f"Loading location memory from {MEMORY_FILE}...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    memory = LocationMemory(save_path=MEMORY_FILE)
    memory.load_memory()
    
    print("Adding/Updating Points of Interest...")
    
    for name, coords in POINTS_OF_INTEREST.items():
        # We give it a high visit_score (e.g., 10) so it's prioritized.
        # We don't save a real image, just a placeholder path.
        dummy_image_path = os.path.join(IMAGE_DIR, f"poi_{name}.png")
        
        memory.locations[coords] = True # Add/update the location
        memory.visit_scores[coords] = 10.0 # Set a high score
        memory.location_images[coords] = dummy_image_path
        
        print(f"  - Added POI: {name} at {coords}")
        
    memory.save_memory()
    print("\n✅ Successfully saved updated memory to {MEMORY_FILE}.")

if __name__ == "__main__":
    main()
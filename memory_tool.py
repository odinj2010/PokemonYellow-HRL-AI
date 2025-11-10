# memory_tool.py
#
# A command-line tool for reading, writing, and searching memory in the Pokemon Yellow game.
#
# --- HOW TO USE ---
#
# Read a memory address:
# python memory_tool.py read 0xD35D
#
# ... more commands to be added ...
#

import argparse
import os
from pyboy import PyBoy

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")

def main():
    parser = argparse.ArgumentParser(description="Command-line memory tool for Pokemon Yellow.")
    parser.add_argument("--state", help="Path to the save state file to load (e.g., states/my_game.state).", default=STATE_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Read Command ---
    parser_read = subparsers.add_parser("read", help="Read a value from a memory address.")
    parser_read.add_argument("address", help="The memory address to read from (e.g., 0xD35D).")

    # --- Write Command ---
    parser_write = subparsers.add_parser("write", help="Write a value to a memory address.")
    parser_write.add_argument("address", help="The memory address to write to (e.g., 0xD35D).")
    parser_write.add_argument("value", help="The value to write (0-255).")

    # --- Search Command ---
    parser_search = subparsers.add_parser("search", help="Search for memory addresses based on values.")
    search_subparsers = parser_search.add_subparsers(dest="search_command", required=True)

    parser_search_equal = search_subparsers.add_parser("equal", help="Search for addresses with a specific value.")
    parser_search_equal.add_argument("value", help="The value to search for (0-255).")

    search_subparsers.add_parser("changed", help="Search for addresses whose values have changed since the last snapshot.")
    search_subparsers.add_parser("unchanged", help="Search for addresses whose values have not changed since the last snapshot.")
    search_subparsers.add_parser("increased", help="Search for addresses whose values have increased since the last snapshot.")
    search_subparsers.add_parser("decreased", help="Search for addresses whose values have decreased since the last snapshot.")
    search_subparsers.add_parser("reset", help="Reset the search, considering all memory addresses.")

    args = parser.parse_args()

    # Use the provided state path, or the default one
    current_state_path = args.state

    if not os.path.exists(current_state_path):
        print(f"Error: Save state not found at {current_state_path}")
        return

    pyboy = PyBoy(ROM_PATH, window="null") # No window needed for this tool
    pyboy.set_emulation_speed(0) # Run as fast as possible

    with open(current_state_path, "rb") as f:
        pyboy.load_state(f)

    # --- Memory Snapshot and Candidates ---
    # These will be stored in a temporary file to persist across calls
    SNAPSHOT_FILE = "memory_snapshot.pkl"
    CANDIDATES_FILE = "memory_candidates.pkl"

    last_snapshot = None
    candidates = set(range(0x0000, 0xFFFF + 1)) # All possible addresses

    if os.path.exists(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE, "rb") as f:
                import pickle
                last_snapshot = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load last snapshot: {e}")
            last_snapshot = None

    if os.path.exists(CANDIDATES_FILE):
        try:
            with open(CANDIDATES_FILE, "rb") as f:
                import pickle
                candidates = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load candidates: {e}")
            candidates = set(range(0x0000, 0xFFFF + 1))

    current_memory = pyboy.memory[0x0000:0xFFFF+1]

    if args.command == "read":
        try:
            address = int(args.address, 16)
            value = pyboy.memory[address]
            print(f"Value at address {args.address}: 0x{value:02X} ({value})")
        except ValueError:
            print(f"Error: Invalid address format '{args.address}'. Please use hexadecimal format (e.g., 0xD35D).")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    elif args.command == "write":
        try:
            address = int(args.address, 16)
            value = int(args.value)
            if not (0 <= value <= 255):
                raise ValueError("Value must be between 0 and 255.")
            
            pyboy.memory[address] = value
            with open(current_state_path, "wb") as f:
                pyboy.save_state(f)
            print(f"Wrote value {value} to address {args.address} and saved the state.")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    elif args.command == "search":
        new_candidates = set()
        if args.search_command == "reset":
            candidates = set(range(0x0000, 0xFFFF + 1))
            last_snapshot = current_memory # Initialize last_snapshot here
            print("Search reset. All memory addresses are now candidates.")
        elif args.search_command == "equal":
            try:
                search_value = int(args.value)
                if not (0 <= search_value <= 255):
                    raise ValueError("Search value must be between 0 and 255.")
                
                for addr in candidates:
                    if current_memory[addr] == search_value:
                        new_candidates.add(addr)
                candidates = new_candidates
                print(f"Found {len(candidates)} candidates with value {search_value}.")
            except ValueError as e:
                print(f"Error: {e}")
        elif args.search_command == "changed":
            if last_snapshot is None:
                print("Error: No previous snapshot. Please run 'search reset' first.")
            else:
                # Debug prints
                print(f"DEBUG: len(last_snapshot) = {len(last_snapshot)}")
                print(f"DEBUG: len(current_memory) = {len(current_memory)}")
                print(f"DEBUG: last_snapshot[0xD35D] = {last_snapshot[0xD35D]}")
                print(f"DEBUG: current_memory[0xD35D] = {current_memory[0xD35D]}")

                for addr in candidates:
                    if current_memory[addr] != last_snapshot[addr]:
                        new_candidates.add(addr)
                candidates = new_candidates
                print(f"Found {len(candidates)} candidates whose values have changed.")
        elif args.search_command == "unchanged":
            if last_snapshot is None: # Add this check here instead
                print("Error: No previous snapshot. Please run 'search reset' first.")
            else:
                for addr in candidates:
                    if current_memory[addr] == last_snapshot[addr]:
                        new_candidates.add(addr)
                candidates = new_candidates
                print(f"Found {len(candidates)} candidates whose values have not changed.")
        elif args.search_command == "increased":
            if last_snapshot is None: # Add this check here instead
                print("Error: No previous snapshot. Please run 'search reset' first.")
            else:
                for addr in candidates:
                    if current_memory[addr] > last_snapshot[addr]:
                        new_candidates.add(addr)
                candidates = new_candidates
                print(f"Found {len(candidates)} candidates whose values have increased.")
        elif args.search_command == "decreased":
            if last_snapshot is None: # Add this check here instead
                print("Error: No previous snapshot. Please run 'search reset' first.")
            else:
                for addr in candidates:
                    if current_memory[addr] < last_snapshot[addr]:
                        new_candidates.add(addr)
                candidates = new_candidates
                print(f"Found {len(candidates)} candidates whose values have decreased.")
        
        # Save current memory as last snapshot for next search
        last_snapshot = current_memory
        
        # Save candidates and snapshot for persistence
        try:
            with open(SNAPSHOT_FILE, "wb") as f:
                import pickle
                pickle.dump(last_snapshot, f)
            with open(CANDIDATES_FILE, "wb") as f:
                import pickle
                pickle.dump(candidates, f)
        except Exception as e:
            print(f"Warning: Could not save snapshot or candidates: {e}")

    pyboy.stop()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

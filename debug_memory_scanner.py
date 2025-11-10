# debug_memory_scanner.py
#
# A GUI-based memory scanner to find unknown memory addresses.
# Runs a Tkinter window alongside the PyBoy game window.
# (FIXED: Added a 'Live Update' checkbox for real-time value watching)

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pyboy import PyBoy

# --- Constants ---
ROM_PATH = "PokemonYellow.gb"
STATE_DIR = "states"
STATE_PATH = os.path.join(STATE_DIR, "new_game.state")

# --- Main Application Class ---
class MemoryScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Scanner")
        self.root.geometry("350x550") # Width x Height (made taller for checkbox)
        self.root.resizable(False, False)

        # --- PyBoy Game Instance ---
        if not os.path.exists(STATE_PATH):
            messagebox.showerror("Error", f"Save state not found: {STATE_PATH}\nPlease create it first.")
            root.destroy()
            return
            
        print(f"Loading ROM: {ROM_PATH}")
        print(f"Loading state: {STATE_PATH}")
        self.pyboy = PyBoy(ROM_PATH, window="SDL2")
        self.pyboy.set_emulation_speed(1)
        with open(STATE_PATH, "rb") as f:
            self.pyboy.load_state(f)
        
        # --- Memory State Variables ---
        self.full_memory_size = 0xFFFF + 1 # 65,536 addresses (0x0000 to 0xFFFF)
        
        # (FIXED) We now need two snapshots:
        # 1. last_snapshot: For filtering (when a button is pressed)
        # 2. last_live_snapshot: For live display (updated every frame)
        self.last_snapshot = None
        self.last_live_snapshot = None 
        
        self.candidates = set(range(self.full_memory_size)) # Start with all addresses
        
        # --- (NEW) GUI Variable ---
        self.live_update_var = tk.BooleanVar(value=False)
        
        # --- GUI Setup ---
        self.setup_gui()
        
        # --- Start Game Loop ---
        self.update_listbox(live=False) # Initial call to show all addresses
        self.game_loop()

    def setup_gui(self):
        # --- Info Label ---
        self.info_label = ttk.Label(self.root, text="Candidates Found: 65536", font=("Arial", 10, "bold"))
        self.info_label.pack(pady=10)

        # --- Initial Scan Button ---
        self.scan_first_btn = ttk.Button(self.root, text="Scan First State (Start New Search)", command=self.scan_first)
        self.scan_first_btn.pack(fill='x', padx=20, pady=5)
        
        # --- Filter Frame ---
        filter_frame = ttk.LabelFrame(self.root, text="Filter Next Scan")
        filter_frame.pack(fill='x', padx=20, pady=10)

        self.scan_changed_btn = ttk.Button(filter_frame, text="Scan Changed", command=self.scan_changed)
        self.scan_changed_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.scan_unchanged_btn = ttk.Button(filter_frame, text="Scan Unchanged", command=self.scan_unchanged)
        self.scan_unchanged_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.scan_increased_btn = ttk.Button(filter_frame, text="Scan Increased", command=self.scan_increased)
        self.scan_increased_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.scan_decreased_btn = ttk.Button(filter_frame, text="Scan Decreased", command=self.scan_decreased)
        self.scan_decreased_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        filter_frame.grid_columnconfigure(0, weight=1)
        filter_frame.grid_columnconfigure(1, weight=1)

        # --- Exact Value Frame ---
        value_frame = ttk.LabelFrame(self.root, text="Exact Value Scan")
        value_frame.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(value_frame, text="Value (int):").pack(side=tk.LEFT, padx=5)
        self.value_entry = ttk.Entry(value_frame, width=10)
        self.value_entry.pack(side=tk.LEFT, padx=5, expand=True, fill='x')
        
        self.scan_value_btn = ttk.Button(value_frame, text="Scan for Value", command=self.scan_value)
        self.scan_value_btn.pack(side=tk.LEFT, padx=5)

        # --- Reset Button ---
        self.reset_btn = ttk.Button(self.root, text="Reset (Show All Addresses)", command=self.reset_search)
        self.reset_btn.pack(fill='x', padx=20, pady=10)
        
        # --- Results Listbox ---
        results_frame = ttk.Frame(self.root)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        ttk.Label(results_frame, text="Results:").pack(anchor='w')
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL)
        self.results_listbox = tk.Listbox(
            results_frame, 
            yscrollcommand=scrollbar.set,
            font=("Courier New", 10) 
        )
        scrollbar.config(command=self.results_listbox.yview)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_listbox.pack(side=tk.LEFT, fill='both', expand=True)
        
        # --- (NEW) Live Update Checkbox ---
        self.live_update_check = ttk.Checkbutton(
            self.root,
            text="Live Update Values (Warning: Slows down if > 1000 results)",
            variable=self.live_update_var
        )
        self.live_update_check.pack(padx=20, pady=5)

    # --- Game Loop ---
    def game_loop(self):
        """ (FIXED) The main loop that ticks PyBoy and handles live updates. """
        try:
            if not self.pyboy.tick():
                self.root.quit() # Stop if game window is closed
                return
            
            # (NEW) Get a live snapshot every single frame
            self.last_live_snapshot = self.get_current_memory()
            
            # (NEW) If the live update box is checked, refresh the list
            if self.live_update_var.get():
                self.update_listbox(live=True)
            
            # Schedule the next game loop call
            self.root.after(1, self.game_loop) # ~1ms delay
        except Exception:
            # Window was likely closed
            pass

    # --- Helper Functions ---
    def get_current_memory(self):
        """ Returns a full snapshot of the game's RAM. """
        return self.pyboy.memory[0x0000:0xFFFF+1] 
        
    def update_listbox(self, live=False):
        """ 
        (FIXED) Updates the GUI listbox.
        - If live=False: Uses 'self.last_snapshot' (from a button press)
        - If live=True:  Uses 'self.last_live_snapshot' (from the game loop)
        """
        self.results_listbox.delete(0, tk.END)
        
        sorted_candidates = sorted(list(self.candidates))
        self.info_label.config(text=f"Candidates Found: {len(self.candidates)}")

        # Determine which snapshot to use
        if live:
            snapshot_to_use = self.last_live_snapshot
            if snapshot_to_use is None: # Game loop hasn't run yet
                snapshot_to_use = self.get_current_memory()
                self.last_live_snapshot = snapshot_to_use
        else:
            snapshot_to_use = self.last_snapshot

        # (NEW) Handle the display logic
        if snapshot_to_use is None:
            # We don't have a snapshot yet (e.g., on init or reset)
            display_limit = min(len(sorted_candidates), 1000)
            if len(sorted_candidates) > 1000:
                self.results_listbox.insert(tk.END, f"Showing first 1000 of {len(sorted_candidates)}...")
            
            for i in range(display_limit):
                addr = sorted_candidates[i]
                self.results_listbox.insert(tk.END, f"0x{addr:04X} (Scan first to see values)")
        else:
            # We have a snapshot, so we can show values
            display_limit = min(len(sorted_candidates), 1000)
            
            if len(sorted_candidates) > 1000 and live:
                self.results_listbox.insert(tk.END, f"Too many results ({len(sorted_candidates)}) for live update.")
                self.results_listbox.insert(tk.END, "Please filter the list below 1000.")
                return # Abort live update if list is too long
            elif len(sorted_candidates) > 1000:
                 self.results_listbox.insert(tk.END, f"Showing first 1000 of {len(sorted_candidates)}...")
                
            for i in range(display_limit):
                addr = sorted_candidates[i]
                value = snapshot_to_use[addr] # Get value from the correct snapshot
                self.results_listbox.insert(tk.END, f"0x{addr:04X} : {value:<3} (0x{value:02X})")

    # --- Button Callbacks (These show STATIC results) ---
    def scan_first(self):
        """ Takes the initial snapshot and updates the list. """
        print("Scanned first state.")
        self.last_snapshot = self.get_current_memory()
        self.candidates = set(range(self.full_memory_size))
        self.update_listbox(live=False) # Show static results of this scan

    def scan_changed(self):
        self.filter_candidates(lambda new, old, addr: new[addr] != old[addr])

    def scan_unchanged(self):
        self.filter_candidates(lambda new, old, addr: new[addr] == old[addr])

    def scan_increased(self):
        self.filter_candidates(lambda new, old, addr: new[addr] > old[addr])

    def scan_decreased(self):
        self.filter_candidates(lambda new, old, addr: new[addr] < old[addr])

    def scan_value(self):
        try:
            value_to_find = int(self.value_entry.get())
            if not (0 <= value_to_find <= 255):
                raise ValueError("Value must be between 0 and 255")
                
            self.filter_candidates(lambda new, old, addr: new[addr] == value_to_find)
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid integer between 0 and 255.\n{e}")

    def filter_candidates(self, filter_func):
        """ Generic function to apply a filter to the candidate list. """
        if self.last_snapshot is None:
            messagebox.showwarning("Warning", "Please press 'Scan First State' to start a new search.")
            return

        new_snapshot = self.get_current_memory()
        new_candidates = set()
        
        for addr in self.candidates:
            if filter_func(new_snapshot, self.last_snapshot, addr):
                new_candidates.add(addr)
                
        self.candidates = new_candidates
        self.last_snapshot = new_snapshot # The new snapshot becomes the "last" one for filtering
        self.update_listbox(live=False) # Show the static results of this filter
        print(f"Filtered. Found {len(self.candidates)} candidates.")

    def reset_search(self):
        print("Search Reset.")
        self.last_snapshot = None
        self.candidates = set(range(self.full_memory_size))
        self.update_listbox(live=False) # Show the reset list

    def on_closing(self):
        """ Handle window close event. """
        print("Closing scanner...")
        if self.pyboy:
            self.pyboy.stop()
        self.root.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    # Check if tkinter is available
    try:
        root = tk.Tk()
    except tk.TclError:
        print("Error: Could not initialize Tkinter.")
        print("If you are on Linux, you may need to install it, e.g.:")
        print("sudo apt-get install python3-tk")
        exit()
        
    app = MemoryScannerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Scanner interrupted.")
        app.on_closing()
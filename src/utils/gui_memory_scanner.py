# src/utils/gui_memory_scanner.py
#
# This is the refactored GUI Memory Scanner, designed to be
# imported and instantiated by another script (like the DebugDashboardWrapper).
# It does NOT create its own PyBoy instance.

import tkinter as tk
from tkinter import ttk, messagebox

class GUIMemoryScanner:
    def __init__(self, root, pyboy_instance):
        """
        Initializes the scanner GUI.
        
        Args:
            root (tk.Tk): The root Tkinter object.
            pyboy_instance (PyBoy): The PyBoy game instance to hook into.
        """
        self.root = root
        self.root.title("Memory Scanner (Live)")
        self.root.geometry("350x550") # Width x Height
        self.root.resizable(False, False)
        
        # This is the shared PyBoy object from the main thread
        self.pyboy = pyboy_instance
        
        # --- Memory State Variables ---
        self.full_memory_size = 0xFFFF + 1 # 65,536 addresses
        self.last_snapshot = None
        self.last_live_snapshot = None 
        self.candidates = set(range(self.full_memory_size))
        
        self.live_update_var = tk.BooleanVar(value=False)
        
        self.setup_gui()
        self.update_listbox(live=False)
        
        # Start the local loop to update the 'live' snapshot
        self.update_live_snapshot()

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
        
        # --- Live Update Checkbox ---
        self.live_update_check = ttk.Checkbutton(
            self.root,
            text="Live Update Values (Warning: Slows if > 1000 results)",
            variable=self.live_update_var
        )
        self.live_update_check.pack(padx=20, pady=5)

    # --- Game Loop (Replaced) ---
    def update_live_snapshot(self):
        """
        This loop runs on the Tkinter thread. It continuously
        grabs the latest memory snapshot from the shared pyboy object.
        """
        try:
            self.last_live_snapshot = self.get_current_memory()
            
            if self.live_update_var.get():
                self.update_listbox(live=True)
            
            # Schedule the next snapshot
            self.root.after(50, self.update_live_snapshot) # Update ~20fps
        except Exception:
            # Main window was likely closed
            pass

    # --- Helper Functions ---
    def get_current_memory(self):
        """ Reads from the shared PyBoy instance's memory. """
        # This is the core connection to the main game thread
        return self.pyboy.memory[0x0000:0xFFFF+1] 
        
    def update_listbox(self, live=False):
        """ Updates the GUI listbox with candidates and values. """
        self.results_listbox.delete(0, tk.END)
        
        sorted_candidates = sorted(list(self.candidates))
        self.info_label.config(text=f"Candidates Found: {len(self.candidates)}")

        if live:
            snapshot_to_use = self.last_live_snapshot
            if snapshot_to_use is None: # Not ready yet
                self.results_listbox.insert(tk.END, "Waiting for live snapshot...")
                return
        else:
            snapshot_to_use = self.last_snapshot

        if snapshot_to_use is None:
            display_limit = min(len(sorted_candidates), 1000)
            if len(sorted_candidates) > 1000:
                self.results_listbox.insert(tk.END, f"Showing first 1000 of {len(sorted_candidates)}...")
            for i in range(display_limit):
                addr = sorted_candidates[i]
                self.results_listbox.insert(tk.END, f"0x{addr:04X} (Scan first to see values)")
        else:
            display_limit = min(len(sorted_candidates), 1000)
            
            if len(sorted_candidates) > 1000 and live:
                self.results_listbox.insert(tk.END, f"Too many results ({len(sorted_candidates)}) for live update.")
                self.results_listbox.insert(tk.END, "Please filter the list below 1000.")
                return
            elif len(sorted_candidates) > 1000:
                 self.results_listbox.insert(tk.END, f"Showing first 1000 of {len(sorted_candidates)}...")
                
            for i in range(display_limit):
                addr = sorted_candidates[i]
                value = snapshot_to_use[addr]
                self.results_listbox.insert(tk.END, f"0x{addr:04X} : {value:<3} (0x{value:02X})")

    # --- Button Callbacks (These show STATIC results) ---
    def scan_first(self):
        print("Scanner: Scanned first state.")
        self.last_snapshot = self.get_current_memory()
        self.candidates = set(range(self.full_memory_size))
        self.update_listbox(live=False)

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
        if self.last_snapshot is None:
            messagebox.showwarning("Warning", "Please press 'Scan First State' to start a new search.")
            return

        new_snapshot = self.get_current_memory()
        new_candidates = set()
        
        for addr in self.candidates:
            if filter_func(new_snapshot, self.last_snapshot, addr):
                new_candidates.add(addr)
                
        self.candidates = new_candidates
        self.last_snapshot = new_snapshot
        self.update_listbox(live=False)
        print(f"Scanner: Filtered. Found {len(self.candidates)} candidates.")

    def reset_search(self):
        print("Scanner: Search Reset.")
        self.last_snapshot = None
        self.candidates = set(range(self.full_memory_size))
        self.update_listbox(live=False)

    def on_closing(self):
        """ Handle window close event. """
        print("Scanner: GUI window closed.")
        self.root.destroy()
# src/utils/location_memory.py

import collections
import pickle
import os
from pathlib import Path

class LocationMemory:
    """
    A memory system to store and manage coordinates of rewarding locations.
    """
    def __init__(self, max_size=100, decay_rate=0.95, save_path='logs/location_memory.pkl'):
        """
        Initializes the LocationMemory.

        Args:
            max_size (int): The maximum number of unique locations to store.
            decay_rate (float): The rate at which visit scores decay over time.
            save_path (str): The path to save/load the memory.
        """
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.save_path = save_path
        # Use an OrderedDict to maintain insertion order and easily discard oldest entries
        self.locations = collections.OrderedDict()
        # Keep track of visit counts to identify frequently unrewarding spots
        self.visit_scores = {}
        self.location_images = {}

    def add_location(self, location_key, image=None, image_path=None):
        """
        Adds a new location to the memory or reinforces an existing one.

        Args:
            location_key (tuple): A tuple representing the location (e.g., (map_id, x, y)).
            image: The image associated with the location.
            image_path (str, optional): The path to an image associated with the location.
        """
        if location_key in self.locations:
            # Move to end to mark as recently used
            self.locations.move_to_end(location_key)
            # Increase score for reinforcement
            self.visit_scores[location_key] = self.visit_scores.get(location_key, 0) + 1
        else:
            if len(self.locations) >= self.max_size:
                # Discard the least recently used location
                oldest_key, _ = self.locations.popitem(last=False)
                if oldest_key in self.visit_scores:
                    del self.visit_scores[oldest_key]
                if oldest_key in self.location_images:
                    # Remove the old image file
                    old_image_path = self.location_images.pop(oldest_key)
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
            
            self.locations[location_key] = True
            self.visit_scores[location_key] = 1
        
        if image is not None and image_path:
            try:
                # Assuming 'image' is a PIL Image object or similar that has a 'save' method
                image.save(image_path)
                self.location_images[location_key] = image_path
                print(f"Saved image to {image_path}")
            except Exception as e:
                print(f"Error saving image to {image_path}: {e}")

    def decay_location(self, location_key):
        """
        Decays the score of a location, potentially removing it if it's not rewarding.

        Args:
            location_key (tuple): The location to decay.
        """
        if location_key in self.visit_scores:
            self.visit_scores[location_key] *= self.decay_rate
            if self.visit_scores[location_key] < 0.1:
                self.remove_location(location_key)
                
    def remove_location(self, location_key):
        """
        Removes a location from memory.

        Args:
            location_key (tuple): The location to remove.
        """
        if location_key in self.locations:
            del self.locations[location_key]
        if location_key in self.visit_scores:
            del self.visit_scores[location_key]
        if location_key in self.location_images:
            image_path = self.location_images.pop(location_key)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed image {image_path}")
        print(f"Removed location {location_key} from memory.")

    def get_locations(self):
        """
        Returns a list of all stored locations.

        Returns:
            list: A list of location tuples.
        """
        return list(self.locations.keys())

    def get_prioritized_locations(self):
        """
        Returns locations sorted by their visit score in descending order.

        Returns:
            list: A sorted list of location tuples.
        """
        return sorted(self.visit_scores.keys(), key=lambda k: self.visit_scores[k], reverse=True)

    def save_memory(self):
        """Saves the location memory to a file."""
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'locations': self.locations,
                'visit_scores': self.visit_scores,
                'location_images': self.location_images
            }, f)
        print(f"Location memory saved to {self.save_path}")

    def load_memory(self):
        """Loads the location memory from a file."""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
                self.locations = data.get('locations', collections.OrderedDict())
                self.visit_scores = data.get('visit_scores', {})
                self.location_images = data.get('location_images', {})
            print(f"Location memory loaded from {self.save_path}")

    def __len__(self):
        return len(self.locations)

    def __str__(self):
        return f"LocationMemory (size={len(self)}/{self.max_size})"


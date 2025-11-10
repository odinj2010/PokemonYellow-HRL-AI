# src/agents/policy.py
# This file defines the neural network architecture (feature extractor)
# for the specialist and manager agents.

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VisionCNN(BaseFeaturesExtractor):
    """
    A custom Convolutional Neural Network (CNN) to extract features from the 
    'vision' observation (the game screen).
    Renamed from 'CustomCNN'.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        # Input shape (C, H, W)
        self.c, self.h, self.w = observation_space.shape
        super().__init__(observation_space, features_dim)
        n_input_channels = self.c
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute the size of the flattened output vector automatically
        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, self.h, self.w)
            n_flatten = self.cnn(dummy).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self._output_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)

class MapCNN(BaseFeaturesExtractor):
    """
    A dedicated CNN to extract features from the 'map' observation (the exploration map).
    It uses a different architecture optimized for a larger, sparse input map.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = 1 # The map is a single channel (grayscale) image
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        
        with th.no_grad():
            dummy = th.zeros(1, n_input_channels, 128, 128) # Map size is 128x128
            n_flatten = self.cnn(dummy).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )
        self._output_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Add a channel dimension (B, 1, H, W) for the Conv2d layer
        observations = observations.unsqueeze(1)
        cnn_output = self.cnn(observations)
        return self.linear(cnn_output)
    
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    The main feature extractor that processes all parts of the multi-modal
    observation space: vision, stats, key_events, map, menu_state, and items.
    It combines two CNNs (vision, map) and four small MLPs.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        # Get the individual observation spaces from the dictionary
        vision_space = observation_space.spaces["vision"]
        stats_space = observation_space.spaces["stats"]
        key_events_space = observation_space.spaces["key_events"]
        map_space = observation_space.spaces["map"]
        menu_space = observation_space.spaces["menu_state"]
        items_space = observation_space.spaces["items"] # (NEW) Get items space
        
        self.use_memorable_location = "memorable_location" in observation_space.spaces
        
        # Define the output feature dimensions for each sub-network
        vision_feat_dim = 128
        
        # (NEW) Stats input dim is now 33, not 31
        stats_input_dim = int(stats_space.shape[0]) # Should be 33
        stats_embed_dim = 128     
           
        key_events_input_dim = int(key_events_space.shape[0])
        key_events_embed_dim = 64
        
        map_feat_dim = 128
        
        menu_input_dim = int(menu_space.shape[0])
        menu_embed_dim = 16 
        
        # (NEW) Define MLP for the item bag
        items_input_dim = int(items_space.shape[0]) # Should be 40
        items_embed_dim = 64

        memorable_location_embed_dim = 0
        if self.use_memorable_location:
            memorable_location_space = observation_space.spaces["memorable_location"]
            memorable_location_input_dim = int(memorable_location_space.shape[0])
            memorable_location_embed_dim = 32
        
        # Calculate the total feature dimension by summing all individual output dimensions
        total_feat_dim = (
            vision_feat_dim + 
            stats_embed_dim +
            key_events_embed_dim + 
            map_feat_dim + 
            menu_embed_dim +
            items_embed_dim + # (NEW)
            memorable_location_embed_dim
        )
        
        # Initialize the parent class with the combined output dimension
        super().__init__(observation_space, features_dim=total_feat_dim)
        self._features_dim = total_feat_dim
        
        # Initialize the individual extractors (CNNs)
        self.vision_extractor = VisionCNN(vision_space, features_dim=vision_feat_dim)
        self.map_extractor = MapCNN(map_space, features_dim=map_feat_dim)
        
        # Define MLPs (Multi-Layer Perceptrons) for the non-image inputs
        self.stats_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stats_input_dim, stats_embed_dim),
            nn.ReLU()
        )
        self.key_events_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(key_events_input_dim, key_events_embed_dim),
            nn.ReLU()
        )
        self.menu_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(menu_input_dim, menu_embed_dim),
            nn.ReLU()
        )
        # (NEW) Item bag MLP
        self.items_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(items_input_dim, items_embed_dim),
            nn.ReLU()
        )
        if self.use_memorable_location:
            self.memorable_location_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(memorable_location_input_dim, memorable_location_embed_dim),
                nn.ReLU()
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass for the combined extractor.
        It processes each observation element and concatenates the resulting features.
        """
        # --- 1. Vision ---
        vision_tensor = observations["vision"]
        if vision_tensor.dtype == th.uint8:
            vision_tensor = vision_tensor.float() / 255.0
        vision_features = self.vision_extractor(vision_tensor)
        
        # --- 2. Stats ---
        stats_tensor = observations["stats"].float()
        stats_features = self.stats_net(stats_tensor)
        
        # --- 3. Key Events ---
        key_events_tensor = observations["key_events"].float()
        key_events_features = self.key_events_net(key_events_tensor)
        
        # --- 4. Map ---
        map_tensor = observations["map"].float() / 255.0
        map_features = self.map_extractor(map_tensor)
        
        # --- 5. Menu State ---
        menu_tensor = observations["menu_state"].float()
        menu_features = self.menu_net(menu_tensor)
        
        # --- 6. (NEW) Items ---
        items_tensor = observations["items"].float()
        items_features = self.items_net(items_tensor)
        
        # --- 7. Concatenation ---
        combined_features = [
            vision_features, 
            stats_features, 
            key_events_features,
            map_features, 
            menu_features,
            items_features # (NEW)
        ]

        # --- 8. Memorable Location (Conditional) ---
        if self.use_memorable_location:
            memorable_location_tensor = observations["memorable_location"].float()
            memorable_location_features = self.memorable_location_net(memorable_location_tensor)
            combined_features.append(memorable_location_features)

        combined = th.cat(combined_features, dim=1)
        
        return combined
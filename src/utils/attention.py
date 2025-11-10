# src/utils/attention.py
# Refactored version of visualize_attention.py
# Implements Grad-CAM for visualizing agent focus.

import numpy as np
import torch as th
import torch.nn.functional as F
import cv2

def get_last_conv_module(module):
    """
    Finds the last 2D convolutional layer (nn.Conv2d) in a PyTorch module.
    This is the target layer for Grad-CAM.
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, th.nn.Conv2d):
            last_conv = m
    return last_conv

def normalize_to_uint8(img):
    """
    Normalizes any image array to a 3-channel, uint8 format (0-255)
    for consistent visualization and saving.
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype == np.uint8:
        pass
    else:
        img = img.astype(np.uint8)
        
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img]*3, axis=2)
    return img

def generate_grad_cam(model, obs, device="cpu", upsample_size=(160, 144)):
    """
    Generates a Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.
    Highlights the regions in the image most influential in the model's decision.
    
    (Renamed from 'grad_cam_for_obs')
    """
    # --- 1. Setup and Preparation ---
    model_device = th.device(device)
    model.policy.to(model_device)
    model.policy.train()
    
    # Prepare the observation batch (add a batch dimension of 1)
    batch = {}
    for k, v in obs.items():
        arr = np.expand_dims(v, axis=0) # Add batch dimension [1, ...]
        t = th.tensor(arr)
        
        # Normalize vision data (uint8 [0, 255] to float [0.0, 1.0])
        if k == "vision" and t.dtype == th.uint8:
            t = t.float() / 255.0
        else:
            if t.dtype != th.float32:
                t = t.float()
        batch[k] = t.to(model_device)
        
    # --- 2. Identify Target Layers ---
    feat_extractor = model.policy.features_extractor
    
    # The 'vision_extractor' is the module we want to hook into
    if not hasattr(feat_extractor, "vision_extractor"):
        raise RuntimeError("Could not find 'vision_extractor' module in features_extractor.")
    
    vision_module = feat_extractor.vision_extractor
    last_conv = get_last_conv_module(vision_module)
    
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in 'vision_extractor' module.")
        
    # --- 3. Hooks for Activation and Gradient Capture ---
    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
        
    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)
    
    # --- 4. Forward and Backward Pass ---
    
    initial_lstm_states = None
    
    if hasattr(model.policy, "lstm_actor"):
        try:
            lstm_module = model.policy.lstm_actor
            hidden_size = lstm_module.hidden_size
            num_layers = lstm_module.num_layers
            batch_size = 1 
            
            h_0 = th.zeros(num_layers, batch_size, hidden_size).to(model_device)
            c_0 = th.zeros(num_layers, batch_size, hidden_size).to(model_device)
            initial_lstm_states = (h_0, c_0)
        except Exception as e:
            raise RuntimeError(f"Failed to manually create LSTM states: {e}")
    
    # 1. Extract features (this triggers the forward hook)
    features = feat_extractor(batch) # Shape: (1, 256) or (1, 528), etc.
    
    # 2. Pass features through the LSTM
    if hasattr(model.policy, "lstm_actor"):
        if initial_lstm_states is None:
             raise RuntimeError("LSTM module exists but initial_lstm_states are None.")
        
        # Reshape for LSTM: (batch, input) -> (seq_len, batch, input)
        features = features.unsqueeze(0)
        
        features, _ = model.policy.lstm_actor(features, initial_lstm_states)
        
        # Reshape after LSTM: (seq_len, batch, hidden) -> (batch, hidden)
        features = features.squeeze(0)

    # In this architecture, the policy_net processes the LSTM output
    if hasattr(model.policy, "mlp_extractor") and hasattr(model.policy.mlp_extractor, "policy_net"):
        features = model.policy.mlp_extractor.policy_net(features)
    
    # 3. Get the action logits (the target for backpropagation)
    action_logits = model.policy.action_net(features)
    
    # Use the logit of the predicted action as the target score
    target_action_idx = int(action_logits.argmax(dim=1)[0].item())
    target_score = action_logits[0, target_action_idx]
            
    # 4. Perform backward pass
    model.policy.zero_grad()
    target_score.backward(retain_graph=True) 
    
    # 5. Remove hooks
    fh.remove()
    bh.remove()
    
    # --- 5. Grad-CAM Calculation ---
    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM: failed to capture activations or gradients.")
        
    # 1. Calculate feature importance weights (Global Average Pooling on gradients)
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # 2. Generate Class Activation Map (CAM)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam) # Apply ReLU
    cam = cam.squeeze().cpu().numpy()
    
    # 3. Normalize the CAM
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = cam * 0.0
        
    # --- 6. Image Overlay ---
    # 4. Resize CAM and create a heatmap
    cam_resized = cv2.resize((cam * 255).astype(np.uint8), (upsample_size[0], upsample_size[1]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # 5. Prepare the original image
    vision = obs["vision"]
    
    # Handle stacked frames (C, H, W) -> select the last frame
    if vision.ndim == 3 and vision.shape[0] > 1:
        img_gray = vision[-1, :, :] # Get last channel (latest frame)
    else:
        img_gray = vision.squeeze()
        
    # Resize original and convert to BGR (OpenCV's default)
    orig_resized = cv2.resize((img_gray * (1 if img_gray.max() <= 1.0 else 255.0)).astype(np.uint8), (upsample_size[0], upsample_size[1]), interpolation=cv2.INTER_NEAREST)
    orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
    
    # 6. Blend the image and heatmap
    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) # Convert back to RGB
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
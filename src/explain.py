import numpy as np
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, head_name="task", class_idx=None):
        """
        Args:
            x: Input tensor (1, 5, 64, 64)
            head_name: 'shot', 'stress', or 'task'
            class_idx: Target class index. If None, uses the predicted class.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        preds = self.model(x)
        
        if head_name == "shot":
            output = preds[0]
        elif head_name == "stress":
            output = preds[1]
        elif head_name == "task":
            output = preds[2]
        else:
            raise ValueError(f"Unknown head: {head_name}")

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass
        target = output[0, class_idx]
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients.data.cpu().numpy()[0] # (C, H, W)
        activations = self.activations.data.cpu().numpy()[0] # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (x.shape[3], x.shape[2])) # Resize to input size
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8) # Normalize
        
        return cam

def generate_occlusion_map(model, x, head_name="task", class_idx=None, window_size=8, stride=4):
    """
    Generates an occlusion map by sliding a zero-mask over the input.
    """
    model.eval()
    with torch.no_grad():
        preds = model(x)
        if head_name == "shot":
            base_out = preds[0]
        elif head_name == "stress":
            base_out = preds[1]
        elif head_name == "task":
            base_out = preds[2]
        
        if class_idx is None:
            class_idx = torch.argmax(base_out)
        
        base_score = base_out[0, class_idx].item()
    
    heatmap = np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
    counts = np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
    
    h, w = x.shape[2], x.shape[3]
    
    for y in range(0, h - window_size + 1, stride):
        for x_pos in range(0, w - window_size + 1, stride):
            x_occ = x.clone()
            x_occ[:, :, y:y+window_size, x_pos:x_pos+window_size] = 0
            
            with torch.no_grad():
                preds_occ = model(x_occ)
                if head_name == "shot":
                    out_occ = preds_occ[0]
                elif head_name == "stress":
                    out_occ = preds_occ[1]
                elif head_name == "task":
                    out_occ = preds_occ[2]
                
                score_occ = out_occ[0, class_idx].item()
                
            diff = base_score - score_occ
            heatmap[y:y+window_size, x_pos:x_pos+window_size] += diff
            counts[y:y+window_size, x_pos:x_pos+window_size] += 1
            
    heatmap = heatmap / (counts + 1e-8)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    return heatmap

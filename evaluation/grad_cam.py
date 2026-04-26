"""
Grad-CAM Visualization for 3D Brain Scans (Problem 20)

Provides interpretability by visualizing which brain regions
influenced the model's prediction for each patient.

Key features:
- 3D Grad-CAM for volumetric brain scans
- Multi-slice visualization (axial, coronal, sagittal)
- Class-specific activation maps
- Overlay on original scan for clinical interpretation
"""

import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D CNNs.
    Shows which brain regions are most important for the prediction.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: Brain3DCNN model instance
            target_layer: Layer to compute CAM for (default: last conv layer)
        """
        if not TORCH_AVAILABLE:
            self.model = None
            return
        
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Get target layer
        if target_layer is None:
            # Default: last conv layer in layer4
            target_layer = model.get_cam_target_layer()
        
        self.target_layer = target_layer
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)
    
    def generate_cam(self, input_volume: np.ndarray, target_class: int = None) -> np.ndarray:
        """
        Generate 3D Class Activation Map.
        
        Args:
            input_volume: (D, H, W) brain scan
            target_class: Class index to visualize (None = predicted class)
        
        Returns:
            cam: (D, H, W) activation map normalized to [0, 1]
        """
        if self.model is None:
            return np.zeros_like(input_volume)
        
        self.model.eval()
        
        # Prepare input: (D,H,W) → (1, 1, D, H, W)
        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(input_volume).float().unsqueeze(0).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(tensor)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros_like(input_volume)
        
        # Get gradients and activations
        gradients = self.gradients[0]     # (C, d, h, w)
        activations = self.activations[0]  # (C, d, h, w)
        
        # Compute weights via global average pooling of gradients
        weights = gradients.mean(dim=(1, 2, 3))  # (C,)
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU — only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to original volume size
        cam_np = cam.cpu().numpy()
        from scipy.ndimage import zoom
        if cam_np.shape != input_volume.shape:
            factors = [s / c for s, c in zip(input_volume.shape, cam_np.shape)]
            cam_np = zoom(cam_np, factors, order=1)
        
        return cam_np.astype(np.float32)
    
    def visualize(self, input_volume: np.ndarray, cam: np.ndarray,
                  predicted_class: str = "", true_class: str = "",
                  save_path: Path = None, n_slices: int = 5):
        """
        Create multi-view visualization of Grad-CAM overlay.
        
        Args:
            input_volume: (D, H, W) original brain scan
            cam: (D, H, W) activation map
            predicted_class: Predicted class name
            true_class: True class name
            save_path: Where to save the figure
            n_slices: Number of slices to show per view
        """
        fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))
        
        views = [
            ("Axial", 0),
            ("Coronal", 1),
            ("Sagittal", 2),
        ]
        
        # Custom colormap for CAM overlay
        cam_cmap = LinearSegmentedColormap.from_list(
            'cam', [(0, 0, 0, 0), (1, 1, 0, 0.3), (1, 0.5, 0, 0.6), (1, 0, 0, 0.8)]
        )
        
        for row_idx, (view_name, axis) in enumerate(views):
            n_total = input_volume.shape[axis]
            slice_indices = np.linspace(n_total * 0.2, n_total * 0.8, n_slices, dtype=int)
            
            for col_idx, slice_idx in enumerate(slice_indices):
                ax = axes[row_idx, col_idx]
                
                # Get slices
                slicing = [slice(None)] * 3
                slicing[axis] = slice_idx
                img_slice = input_volume[tuple(slicing)]
                cam_slice = cam[tuple(slicing)]
                
                # Display brain scan
                ax.imshow(img_slice.T if axis != 0 else img_slice,
                         cmap='gray', aspect='auto')
                
                # Overlay CAM
                ax.imshow(cam_slice.T if axis != 0 else cam_slice,
                         cmap=cam_cmap, aspect='auto', alpha=0.6)
                
                ax.set_title(f"{view_name} #{slice_idx}", fontsize=9)
                ax.axis('off')
        
        # Overall title
        title = "Grad-CAM: Brain Regions Influencing Prediction"
        if predicted_class:
            title += f"\nPredicted: {predicted_class}"
        if true_class:
            title += f" | True: {true_class}"
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        return fig


def generate_gradcam_visualizations(model, volumes: list, labels: list,
                                     predictions: list, label_names: list,
                                     save_dir: Path, n_samples: int = 5):
    """
    Generate Grad-CAM visualizations for a set of samples.
    
    Args:
        model: Trained Brain3DCNN model
        volumes: List of (D, H, W) brain volumes
        labels: True class indices
        predictions: Predicted class indices
        label_names: ["CN", "MCI", "AD"]
        save_dir: Directory to save visualizations
        n_samples: Number of samples to visualize
    """
    if not TORCH_AVAILABLE or model is None:
        print("  Grad-CAM skipped: PyTorch or model not available")
        return
    
    save_dir = Path(save_dir) / "gradcam"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    gradcam = GradCAM3D(model)
    
    n_samples = min(n_samples, len(volumes))
    
    print(f"\n  Generating Grad-CAM visualizations ({n_samples} samples)...")
    
    for i in range(n_samples):
        vol = volumes[i]
        true_cls = label_names[labels[i]]
        pred_cls = label_names[predictions[i]]
        
        # Generate CAM for predicted class
        cam = gradcam.generate_cam(vol, target_class=predictions[i])
        
        # Save visualization
        save_path = save_dir / f"gradcam_sample_{i+1}_{true_cls}_pred_{pred_cls}.png"
        gradcam.visualize(vol, cam,
                         predicted_class=pred_cls,
                         true_class=true_cls,
                         save_path=save_path)
        
        status = "✓" if labels[i] == predictions[i] else "✗"
        print(f"    [{status}] Sample {i+1}: True={true_cls}, Pred={pred_cls}")
    
    print(f"  Grad-CAM visualizations saved to: {save_dir}")

"""
Stage 2: CNN Model Architecture
================================
Simple yet effective CNN for wafer defect classification.
Supports Grad-CAM via gradient hooks on the final convolutional layer.

Run: python src/model.py  (imports only, no execution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferCNN(nn.Module):
    """
    Convolutional Neural Network for wafer defect classification.
    
    Architecture:
        4 Conv blocks (Conv2d → BatchNorm → ReLU → MaxPool)
        1 Global Average Pooling
        1 Fully Connected classifier
    
    Grad-CAM: The last convolutional layer's activations and gradients
    are captured for visualization.
    """

    def __init__(self, num_classes=8):
        super(WaferCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 8x8 → 4x4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Index of the last Conv2d layer in features (for Grad-CAM)
        self.last_conv_idx = 12  # features[12] = Conv2d(128, 128, 3, padding=1)

        # Register hooks for Grad-CAM on the LAST CONV layer (not the last MaxPool)
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _register_gradcam_hooks(self):
        """Register forward/backward hooks on the LAST CONV layer for Grad-CAM.
        
        The last conv layer is at index self.last_conv_idx (not features[-1]
        which is MaxPool2d). Grad-CAM requires gradients w.r.t. the conv output,
        not the pooling output.
        """
        def save_activation(module, input, output):
            self.activations = output

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Attach hooks to the last Conv2d layer
        last_conv = self.features[self.last_conv_idx]
        last_conv.register_forward_hook(save_activation)
        last_conv.register_full_backward_hook(save_gradient)

    def _initialize_weights(self):
        """Initialize convolutional and linear weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, 64, 64)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        features = self.features(x)
        # self.activations and self.gradients are set by the forward/backward hooks
        # registered on the last Conv2d layer in _register_gradcam_hooks()
        
        pooled = self.global_pool(features).flatten(1)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)
        return output

    def get_gradcam_cam(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap for a given input.
        
        Args:
            x: Input tensor (1, 1, 64, 64)
            class_idx: Target class index (default: predicted class)
        
        Returns:
            cam: Normalized Grad-CAM heatmap (H, W)
            class_idx: Index of the target class
            probs: Softmax probabilities for all classes
        """
        self.eval()
        x = x.clone().requires_grad_(True)
        
        output = self.forward(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for the target class
        self.zero_grad()
        output[0, class_idx].backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0].cpu().detach().numpy()  # (C, h, w)
        activations = self.activations[0].cpu().detach().numpy()  # (C, h, w)
        
        # Global average pooling of gradients → channel weights
        weights = gradients.mean(axis=(1, 2))  # (C,)
        
        # Weighted combination of feature maps
        cam = np.einsum("c,chw->hw", weights, activations)
        cam = np.maximum(cam, 0)  # ReLU to keep only positive contributions
        
        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        
        # Compute softmax probabilities
        probs = F.softmax(output, dim=1)[0].cpu().detach().numpy()
        
        return cam, class_idx, probs


if __name__ == "__main__":
    # Quick model sanity check
    model = WaferCNN(num_classes=8)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Model sanity check passed.")

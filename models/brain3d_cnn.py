"""
True 3D CNN for Brain Volume Feature Extraction

This module implements a 3D Convolutional Neural Network specifically designed
for processing 64×64×64 brain MRI/PET volumes. Unlike 2D approaches that only
look at slices, this processes the entire 3D volume to capture spatial relationships.

Architecture: 4 Conv3D blocks + Global Average Pooling + Feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Brain3DCNN(nn.Module):
    """
    3D CNN for volumetric brain scan analysis.
    
    Architecture:
        Input: (batch, 1, 64, 64, 64)
        Conv3D Block 1: 1 → 32 channels
        Conv3D Block 2: 32 → 64 channels
        Conv3D Block 3: 64 → 128 channels
        Conv3D Block 4: 128 → 256 channels
        Global Average Pooling
        Output: (batch, 256) feature vector
    
    Args:
        in_channels: Number of input channels (1 for grayscale brain scans)
        num_classes: Number of output classes (3 for CN/MCI/AD)
        feature_dim: Dimension of feature vector (default: 256)
        dropout_rate: Dropout probability (default: 0.5)
    """
    
    def __init__(self, in_channels=1, num_classes=3, feature_dim=256, dropout_rate=0.5):
        super(Brain3DCNN, self).__init__()
        
        self.feature_dim = feature_dim
        
        # ── Conv Block 1: 1 → 32 channels ──────────────────────────────
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        # ── Conv Block 2: 32 → 64 channels ─────────────────────────────
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        # ── Conv Block 3: 64 → 128 channels ────────────────────────────
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        # ── Conv Block 4: 128 → 256 channels ───────────────────────────
        self.conv4 = nn.Conv3d(128, feature_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(feature_dim)
        
        # ── Pooling and Regularization ─────────────────────────────────
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.gap = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        
        # ── Classification Head (optional) ─────────────────────────────
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x: (batch, 1, 64, 64, 64) - Input brain volumes
        
        Returns:
            logits: (batch, num_classes) - Class logits
        """
        features = self.extract_features(x)
        
        # Classification head
        x = self.dropout_fc(features)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
    
    def extract_features(self, x):
        """
        Extract feature vector from input volume.
        
        This is the method used for feature extraction in the pipeline.
        
        Args:
            x: (batch, 1, 64, 64, 64) - Input brain volumes
        
        Returns:
            features: (batch, feature_dim) - Feature vectors
        """
        # Input: (batch, 1, 64, 64, 64)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 32, 32, 32, 32)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 64, 16, 16, 16)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 128, 8, 8, 8)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 256, 4, 4, 4)
        
        # Global Average Pooling
        x = self.gap(x)  # (batch, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        return x


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with skip connection.
    Helps with gradient flow in deeper networks.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImprovedBrain3DCNN(nn.Module):
    """
    Improved 3D CNN with residual connections for better gradient flow.
    Use this if the basic Brain3DCNN doesn't perform well.
    """
    
    def __init__(self, in_channels=1, num_classes=3, feature_dim=512, dropout_rate=0.5):
        super(ImprovedBrain3DCNN, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, feature_dim, num_blocks=2, stride=2)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.extract_features(x)
        x = self.dropout(features)
        x = self.fc(x)
        return x
    
    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test the model
    print("Testing Brain3DCNN...")
    
    model = Brain3DCNN(in_channels=1, num_classes=3, feature_dim=256)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 64, 64, 64)
    
    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")  # Should be (2, 256)
    
    # Test classification
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")  # Should be (2, 3)
    
    print("\n✓ Model test passed!")
    
    # Test improved model
    print("\nTesting ImprovedBrain3DCNN...")
    improved_model = ImprovedBrain3DCNN(in_channels=1, num_classes=3, feature_dim=512)
    n_params_improved = sum(p.numel() for p in improved_model.parameters())
    print(f"Total parameters: {n_params_improved:,}")
    
    features_improved = improved_model.extract_features(dummy_input)
    print(f"Feature shape: {features_improved.shape}")  # Should be (2, 512)
    
    print("\n✓ Improved model test passed!")

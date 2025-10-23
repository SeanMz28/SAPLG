# generator.py
"""
Style-Aware Generator and Discriminator for Spelunky level generation.

This module implements the GAN architecture for generating platformer levels
conditioned on structural style vectors.
"""

import torch
import torch.nn as nn


class StyleAwareGenerator(nn.Module):
    """
    Generator that takes style vector as input and outputs level layout.
    
    Based on the research proposal Section 3.2:
    - Input: Style vector s (6-dimensional: room_count, branching, 
             linearity, dead_end_rate, loop_complexity, segment_variance)
    - Output: Level layout L (grid of tiles)
    
    Args:
        style_dim (int): Dimension of style vector (default: 6)
        noise_dim (int): Dimension of random noise vector (default: 100)
        output_height (int): Height of generated level (default: 32)
        output_width (int): Width of generated level (default: 32)
        num_tiles (int): Number of tile types (default: 5 for 0,1,D,E,L)
        hidden_dim (int): Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        style_dim=6,
        noise_dim=100,
        output_height=32,
        output_width=32,
        num_tiles=5,
        hidden_dim=256
    ):
        super().__init__()
        
        self.style_dim = style_dim
        self.noise_dim = noise_dim
        self.output_shape = (output_height, output_width)
        self.num_tiles = num_tiles
        
        # Style encoder: transform style vector to latent space
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Combine style + noise
        combined_dim = hidden_dim + noise_dim
        
        # Main generator network
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256 * 8 * 8),
            nn.ReLU()
        )
        
        # Convolutional upsampling
        self.conv_blocks = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final layer: output tile probabilities
            nn.Conv2d(64, num_tiles, 3, 1, 1),
        )
        
    def forward(self, style_vector, noise=None):
        """
        Forward pass of generator.
        
        Args:
            style_vector (torch.Tensor): Style features, shape (batch, 6)
            noise (torch.Tensor, optional): Random noise, shape (batch, 100)
        
        Returns:
            torch.Tensor: Generated level logits, shape (batch, num_tiles, height, width)
        """
        batch_size = style_vector.size(0)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=style_vector.device)
        
        # Encode style
        style_encoded = self.style_encoder(style_vector)  # (batch, hidden_dim)
        
        # Combine style + noise
        combined = torch.cat([style_encoded, noise], dim=1)
        
        # Generate via fully connected + conv layers
        x = self.fc(combined)
        x = x.view(batch_size, 256, 8, 8)
        levels = self.conv_blocks(x)
        
        return levels


class LevelDiscriminator(nn.Module):
    """
    Discriminator that judges if a level is real or fake.
    Also estimates the style vector (for style consistency loss).
    
    This dual-head architecture enables:
    1. Adversarial training (real/fake classification)
    2. Style consistency enforcement (style vector prediction)
    
    Args:
        input_height (int): Height of input level (default: 32)
        input_width (int): Width of input level (default: 32)
        num_tiles (int): Number of tile types (default: 5)
        style_dim (int): Dimension of style vector (default: 6)
    """
    
    def __init__(
        self,
        input_height=32,
        input_width=32,
        num_tiles=5,
        style_dim=6
    ):
        super().__init__()
        
        self.input_shape = (input_height, input_width)
        self.num_tiles = num_tiles
        self.style_dim = style_dim
        
        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(num_tiles, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Real/fake classification head
        self.fc_realfake = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Style estimation head (for style consistency)
        self.fc_style = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, style_dim)
        )
        
    def forward(self, levels):
        """
        Forward pass of discriminator.
        
        Args:
            levels (torch.Tensor): Level layouts, shape (batch, num_tiles, height, width)
        
        Returns:
            tuple: (real_fake_scores, estimated_style)
                - real_fake_scores: Probability of being real, shape (batch, 1)
                - estimated_style: Predicted style vector, shape (batch, style_dim)
        """
        features = self.conv_blocks(levels)
        features_flat = features.view(features.size(0), -1)
        
        real_fake = self.fc_realfake(features_flat)
        style_pred = self.fc_style(features_flat)
        
        return real_fake, style_pred


# Test the models
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    generator = StyleAwareGenerator().to(device)
    discriminator = LevelDiscriminator().to(device)
    
    print(f"\n🎮 Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"🔍 Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    style_vector = torch.randn(batch_size, 6).to(device)
    
    print(f"\n📊 Testing forward pass with batch_size={batch_size}")
    
    # Generate fake level
    fake_levels = generator(style_vector)
    print(f"✅ Generated levels shape: {fake_levels.shape}")  # (4, 5, 32, 32)
    
    # Discriminate
    real_fake_scores, style_pred = discriminator(fake_levels)
    print(f"✅ Real/fake scores shape: {real_fake_scores.shape}")   # (4, 1)
    print(f"✅ Style predictions shape: {style_pred.shape}")        # (4, 6)
    
    print(f"\n✨ Model test completed successfully!"}
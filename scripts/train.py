# train.py
"""
Training pipeline for Style-Aware GAN level generator.

This script implements:
1. Dataset loading for Spelunky levels with style vectors
2. GAN training loop with adversarial + style consistency losses
3. Checkpoint saving and visualization
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.generators.generator import StyleAwareGenerator, LevelDiscriminator


class SpelunkyDataset(Dataset):
    """Dataset loader for Spelunky levels with their structural style vectors."""
    
    def __init__(self, levels_dir, metrics_file, tile_to_idx):
        self.levels_dir = Path(levels_dir)
        self.tile_to_idx = tile_to_idx
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        # Filter for successful extractions
        self.metrics = {
            k: v for k, v in all_metrics.items() 
            if v.get('success', False)
        }
        
        # Get corresponding level files
        self.level_files = [
            f for f in self.levels_dir.glob("*.txt")
            if f.stem in self.metrics
        ]
        
        print(f"[DIR] Loaded {len(self.level_files)} valid levels from {levels_dir}")
    
    def __len__(self):
        return len(self.level_files)
    
    def __getitem__(self, idx):
        level_file = self.level_files[idx]
        level_name = level_file.stem
        
        # Load level
        with open(level_file, 'r') as f:
            rows = [line.strip() for line in f]
        
        # Convert to tensor (one-hot encoding)
        level_tensor = self._level_to_tensor(rows)
        
        # Get style vector
        metrics = self.metrics[level_name]
        style_vector = torch.tensor([
            metrics['room_count'],
            metrics['branching'],
            metrics['linearity'],
            metrics['dead_end_rate'],
            metrics['loop_complexity'],
            metrics['segment_size_variance']
        ], dtype=torch.float32)
        
        return level_tensor, style_vector, level_name
    
    def _level_to_tensor(self, rows):
        """Convert level from text format to one-hot tensor."""
        H, W = len(rows), len(rows[0])
        num_tiles = len(self.tile_to_idx)
        
        tensor = torch.zeros(num_tiles, H, W)
        for i, row in enumerate(rows):
            for j, char in enumerate(row):
                tile_idx = self.tile_to_idx.get(char, 0)
                tensor[tile_idx, i, j] = 1.0
        
        return tensor


def train_gan(
    generator,
    discriminator,
    dataloader,
    num_epochs=1000,
    device='cuda',
    lr_g=0.0002,
    lr_d=0.0001,
    style_loss_weight=0.01,
    checkpoint_dir='checkpoints',
    save_interval=50
):
    """Train the Style-Aware GAN."""
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    history = {
        'd_loss': [], 'g_loss': [], 'd_loss_real': [],
        'd_loss_fake': [], 'd_style_loss': [],
        'g_loss_adv': [], 'g_loss_style': []
    }
    
    print(f"\n Starting training for {num_epochs} epochs")
    print(f"[STATS] Dataset size: {len(dataloader.dataset)}")
    
    for epoch in range(num_epochs):
        epoch_metrics = {k: [] for k in history.keys()}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (real_levels, style_vectors, _) in enumerate(pbar):
            batch_size = real_levels.size(0)
            real_levels = real_levels.to(device)
            style_vectors = style_vectors.to(device)
            
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Train Discriminator
            opt_d.zero_grad()
            real_scores, real_style_pred = discriminator(real_levels)
            d_loss_real = bce_loss(real_scores, real_labels)
            
            noise = torch.randn(batch_size, 100, device=device)
            fake_levels = generator(style_vectors, noise)
            fake_scores, fake_style_pred = discriminator(fake_levels.detach())
            d_loss_fake = bce_loss(fake_scores, fake_labels)
            d_style_loss = mse_loss(real_style_pred, style_vectors)
            
            d_loss = d_loss_real + d_loss_fake + d_style_loss
            d_loss.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            noise = torch.randn(batch_size, 100, device=device)
            fake_levels = generator(style_vectors, noise)
            fake_scores, fake_style_pred = discriminator(fake_levels)
            
            g_loss_adv = bce_loss(fake_scores, real_labels)
            g_loss_style = mse_loss(fake_style_pred, style_vectors)
            g_loss = g_loss_adv + style_loss_weight * g_loss_style
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

            opt_g.step()
            
            # Record metrics
            epoch_metrics['d_loss'].append(d_loss.item())
            epoch_metrics['g_loss'].append(g_loss.item())
            epoch_metrics['d_loss_real'].append(d_loss_real.item())
            epoch_metrics['d_loss_fake'].append(d_loss_fake.item())
            epoch_metrics['d_style_loss'].append(d_style_loss.item())
            epoch_metrics['g_loss_adv'].append(g_loss_adv.item())
            epoch_metrics['g_loss_style'].append(g_loss_style.item())
            
            pbar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}"
            })
        
        for key in history.keys():
            history[key].append(np.mean(epoch_metrics[key]))
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"D_loss: {history['d_loss'][-1]:.4f} | "
              f"G_loss: {history['g_loss'][-1]:.4f}")
        
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'history': history
            }
            torch.save(checkpoint, checkpoint_path / f'epoch_{epoch+1:04d}.pt')
            print(f"💾 Saved checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    tile_to_idx = {'0': 0, '1': 1, 'D': 2, 'E': 3, 'L': 4}
    
    dataset = SpelunkyDataset(
        levels_dir='captured_levels',
        metrics_file='spelunky_metrics.json',
        tile_to_idx=tile_to_idx
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    generator = StyleAwareGenerator().to(device)
    discriminator = LevelDiscriminator().to(device)
    
    train_gan(generator, discriminator, dataloader, 
              num_epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
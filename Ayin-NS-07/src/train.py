import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.policy import MultiModalSequenceTransformer

class EpisodeSequenceDataset(Dataset):
    def __init__(self, low_dim, vision, tactile, prim, finger, force, direction, returns, episode_lengths, ep_indices, T=6):
        self.low_dim = low_dim
        self.vision = vision
        self.tactile = tactile
        self.prim = prim
        self.finger = finger
        self.force = force
        self.direction = direction
        self.returns = returns
        self.T = T
        
        self.mapping = []
        
        ep_start_indices = [0]
        for length in episode_lengths:
            ep_start_indices.append(ep_start_indices[-1] + length)
            
        for ep_idx in ep_indices:
            start_idx = ep_start_indices[ep_idx]
            length = episode_lengths[ep_idx]
            for step_idx in range(length):
                self.mapping.append((ep_idx, start_idx, step_idx))
                
    def __len__(self):
        return len(self.mapping)
        
    def __getitem__(self, idx):
        ep_idx, ep_start_idx, step_idx = self.mapping[idx]
        
        seq_ld = []
        seq_vis = []
        seq_tac = []
        
        for s in range(step_idx - self.T + 1, step_idx + 1):
            if s < 0:
                seq_ld.append(np.zeros(14, dtype=np.float32))
                seq_vis.append(np.zeros((6, 16, 16), dtype=np.float32))
                seq_tac.append(np.zeros((7, 8, 8), dtype=np.float32))
            else:
                g_idx = ep_start_idx + s
                seq_ld.append(self.low_dim[g_idx])
                seq_vis.append(self.vision[g_idx].astype(np.float32) / 255.0)
                seq_tac.append(self.tactile[g_idx].astype(np.float32) / 255.0)
                
        seq_ld = np.stack(seq_ld)
        seq_vis = np.stack(seq_vis)
        seq_tac = np.stack(seq_tac)
        
        g_idx = ep_start_idx + step_idx
        
        return {
            "vis_grid": torch.tensor(seq_vis, dtype=torch.float32),
            "tac_image": torch.tensor(seq_tac, dtype=torch.float32),
            "low_dim": torch.tensor(seq_ld, dtype=torch.float32),
            "prim": torch.tensor(self.prim[g_idx], dtype=torch.long),
            "finger": torch.tensor(self.finger[g_idx], dtype=torch.long),
            "force": torch.tensor(self.force[g_idx], dtype=torch.float32),
            "direction": torch.tensor(self.direction[g_idx], dtype=torch.float32),
            "return": torch.tensor(self.returns[g_idx], dtype=torch.float32)
        }

def train_epoch(model, loader, optimizer, device, tau=0.15):
    model.train()
    total_loss = 0
    total_c_loss = 0
    total_a_loss = 0
    
    for batch in tqdm(loader, desc="Training Batches", leave=False):
        vis = batch["vis_grid"].to(device)
        tac = batch["tac_image"].to(device)
        ld = batch["low_dim"].to(device)
        prim = batch["prim"].to(device)
        finger = batch["finger"].to(device)
        force = batch["force"].to(device)
        direction = batch["direction"].to(device)
        returns = batch["return"].to(device)
        
        optimizer.zero_grad()
        
        prim_logits, finger_logits, force_pred, dir_pred, val_pred = model(vis, tac, ld)
        
        # Critic Value Loss
        loss_critic = F.mse_loss(val_pred, returns)
        
        # AWR weights based on advantage (return - value)
        with torch.no_grad():
            advantage = returns - val_pred
            weights = torch.exp(advantage / tau).clamp(max=20.0)
            weights = weights / (weights.mean() + 1e-8)
            
        # Actor losses
        loss_prim = (F.cross_entropy(prim_logits, prim, reduction='none') * weights).mean()
        loss_finger = (F.cross_entropy(finger_logits, finger, reduction='none') * weights).mean()
        loss_force = (F.mse_loss(force_pred, force, reduction='none') * weights).mean()
        loss_dir = (F.mse_loss(dir_pred, direction, reduction='none').sum(dim=-1) * weights).mean()
        
        loss_actor = loss_prim + loss_finger + loss_force + loss_dir
        loss = loss_critic + loss_actor
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_c_loss += loss_critic.item()
        total_a_loss += loss_actor.item()
        
    n = len(loader)
    return total_loss / n, total_c_loss / n, total_a_loss / n

def validate(model, loader, device, tau=0.15):
    model.eval()
    total_loss = 0
    total_c_loss = 0
    total_a_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            vis = batch["vis_grid"].to(device)
            tac = batch["tac_image"].to(device)
            ld = batch["low_dim"].to(device)
            prim = batch["prim"].to(device)
            finger = batch["finger"].to(device)
            force = batch["force"].to(device)
            direction = batch["direction"].to(device)
            returns = batch["return"].to(device)
            
            prim_logits, finger_logits, force_pred, dir_pred, val_pred = model(vis, tac, ld)
            
            loss_critic = F.mse_loss(val_pred, returns)
            
            advantage = returns - val_pred
            weights = torch.exp(advantage / tau).clamp(max=20.0)
            weights = weights / (weights.mean() + 1e-8)
            
            loss_prim = (F.cross_entropy(prim_logits, prim, reduction='none') * weights).mean()
            loss_finger = (F.cross_entropy(finger_logits, finger, reduction='none') * weights).mean()
            loss_force = (F.mse_loss(force_pred, force, reduction='none') * weights).mean()
            loss_dir = (F.mse_loss(dir_pred, direction, reduction='none').sum(dim=-1) * weights).mean()
            
            loss_actor = loss_prim + loss_finger + loss_force + loss_dir
            loss = loss_critic + loss_actor
            
            total_loss += loss.item()
            total_c_loss += loss_critic.item()
            total_a_loss += loss_actor.item()
            
    n = len(loader)
    return total_loss / n, total_c_loss / n, total_a_loss / n

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.15)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_path = ROOT / "reconstructed_dataset.npz"
    if not dataset_path.exists():
        print(f"Error: dataset file {dataset_path} not found. Please run reconstruct_dataset.py first.")
        sys.exit(1)
        
    print("Loading npz dataset...")
    data = np.load(dataset_path)
    low_dim = data["low_dim_states"]
    vision = data["vision_grids"]
    tactile = data["tactile_images"]
    prim = data["prim_labels"]
    finger = data["finger_labels"]
    force = data["forces"]
    direction = data["directions"]
    returns = data["returns"]
    episode_lengths = data["episode_lengths"]
    
    num_episodes = len(episode_lengths)
    print(f"Total episodes: {num_episodes}")
    
    # Train-val split by episode indices
    np.random.seed(42)
    shuffled_episodes = np.random.permutation(num_episodes)
    val_split = int(0.10 * num_episodes)
    val_eps = shuffled_episodes[:val_split]
    train_eps = shuffled_episodes[val_split:]
    
    print(f"Training on {len(train_eps)} episodes, validating on {len(val_eps)} episodes.")
    
    train_dataset = EpisodeSequenceDataset(
        low_dim, vision, tactile, prim, finger, force, direction, returns, episode_lengths, train_eps, T=6
    )
    val_dataset = EpisodeSequenceDataset(
        low_dim, vision, tactile, prim, finger, force, direction, returns, episode_lengths, val_eps, T=6
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = MultiModalSequenceTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "policy.pth"
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_c, tr_a = train_epoch(model, train_loader, optimizer, device, tau=args.tau)
        val_loss, val_c, val_a = validate(model, val_loader, device, tau=args.tau)
        
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} (Critic: {tr_c:.4f}, Actor: {tr_a:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Critic: {val_c:.4f}, Actor: {val_a:.4f})")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model weights to {save_path}")
            
    print("Training finished successfully!")

if __name__ == "__main__":
    main()

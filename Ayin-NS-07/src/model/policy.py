import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalSequenceTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_primitives=10, num_fingers=7, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Vision Grid Encoder: Input is (batch, 6, 16, 16)
        self.vis_cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 8, 8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 4, 4)
            nn.Flatten(),     # -> 512
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # 2. Tactile Image Encoder: Input is (batch, 7, 8, 8)
        self.tac_cnn = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 4, 4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),     # -> 512
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # 3. Low-Dim State Encoder: Input is (batch, 14)
        self.ld_mlp = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU()
        )
        
        # 4. Token Mapper: Fuses vis (128) + tac (128) + ld (64) = 320 -> embed_dim (128)
        self.token_map = nn.Sequential(
            nn.Linear(320, embed_dim),
            nn.ReLU()
        )
        
        # 5. Learned Temporal Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, embed_dim))
        
        # 6. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 7. Actor Heads (Outputs from final representation of last step)
        self.primitive_head = nn.Linear(embed_dim, num_primitives)
        self.finger_head = nn.Linear(embed_dim, num_fingers)
        self.force_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.direction_head = nn.Linear(embed_dim, 2)
        
        # 8. Critic Head
        self.value_head = nn.Linear(embed_dim, 1)
        
    def forward(self, vis_grid, tac_image, low_dim):
        # Input shapes:
        # vis_grid: (batch_size, seq_len, 6, 16, 16)
        # tac_image: (batch_size, seq_len, 7, 8, 8)
        # low_dim: (batch_size, seq_len, 14)
        
        batch_size, seq_len, _, _, _ = vis_grid.shape
        
        # Flatten temporal dimension to pass through CNN encoders
        vis_flat = vis_grid.view(-1, 6, 16, 16)
        tac_flat = tac_image.view(-1, 7, 8, 8)
        ld_flat = low_dim.view(-1, 14)
        
        # Extract embeddings
        vis_emb = self.vis_cnn(vis_flat).view(batch_size, seq_len, 128)
        tac_emb = self.tac_cnn(tac_flat).view(batch_size, seq_len, 128)
        ld_emb = self.ld_mlp(ld_flat).view(batch_size, seq_len, 64)
        
        # Concatenate and map to token representation
        fused = torch.cat([vis_emb, tac_emb, ld_emb], dim=-1)
        tokens = self.token_map(fused)
        
        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :seq_len, :]
        
        # Pass through Transformer Encoder
        # Out shape: (batch_size, seq_len, embed_dim)
        tokens_out = self.transformer(tokens)
        
        # We take the representation of the last token in the sequence (current step)
        rep = tokens_out[:, -1, :]
        
        # Outputs
        prim_logits = self.primitive_head(rep)
        finger_logits = self.finger_head(rep)
        force = self.force_head(rep).squeeze(-1)
        direction = self.direction_head(rep)
        
        # Normalize direction vector to unit sphere
        dir_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-8
        direction = direction / dir_norm
        
        value = self.value_head(rep).squeeze(-1)
        
        return prim_logits, finger_logits, force, direction, value

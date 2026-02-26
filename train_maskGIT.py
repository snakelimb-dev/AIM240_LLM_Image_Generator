import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from pathlib import Path
from tqdm import tqdm
import math
import clip

from lfq_vqvae import LFQ_VQVAE
from pixelclip import PixelCLIP

CONFIG = {
    "cache_file": "maskgit_gbc_cache.pt",
    "checkpoint_file": "maskgit_gbc_checkpoint.pt",
    "output_file": "maskgit_gbc_best.pt",
    "vis_folder": "maskgit_visuals",
    "vqvae_weights": "lfq_gan_vqvae_best.pt",
    "pixelclip_weights": "pixelclip_best.pt",
    "epochs": 300,
    "batch_size": 64, 
    "lr": 4e-4,
    "seq_len": 1024,
    "codebook_size": 4096, 
    "embed_dim": 12,
    "clip_dim": 512,
    "transformer_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "cfg_scale": 4.5,
    "temp": 1.0,
    "test_prompts": [
        "a vibrant red apple, pixel art",
        "a bright yellow banana, pixel art",
        "a crisp green pear, pixel art"
    ]
}

class MaskGITDataset(Dataset):
    def __init__(self, cache_path):
        self.cache = torch.load(cache_path, map_location="cpu")
        self.keys = list(self.cache.keys())
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        d = self.cache[self.keys[idx]]
        return d["tokens"].view(-1).long(), d["clip_emb"].float()

class MaskGIT(nn.Module):
    def __init__(self, codebook_size, seq_len, clip_dim, embed_dim, n_layers, n_heads):
        super().__init__()
        self.mask_id = codebook_size 
        self.token_emb = nn.Embedding(codebook_size + 1, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.null_cond = nn.Parameter(torch.randn(clip_dim) * 0.02)
        self.cond_proj = nn.Linear(clip_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, tokens, cond_emb):
        x = self.token_emb(tokens) + self.pos_emb
        c = self.cond_proj(cond_emb).unsqueeze(1)
        x = torch.cat([c, x], dim=1)
        x = self.transformer(x)
        x = self.ln_f(x[:, 1:, :])
        return F.linear(x, self.token_emb.weight[:-1])

def cosine_schedule(t): return math.cos(t * math.pi * 0.5)

@torch.no_grad()
def visualize(model, vqvae, pixelclip, text_enc, device, epoch):
    model.eval()
    torch.cuda.empty_cache() # Clear VRAM before inference
    txt_tokens = clip.tokenize(CONFIG["test_prompts"]).to(device)
    raw_emb = F.normalize(text_enc.encode_text(txt_tokens).float(), dim=-1)
    cond_emb = pixelclip.encode_text_raw(raw_emb)
    b, n = len(CONFIG["test_prompts"]), CONFIG["seq_len"]
    tokens = torch.full((b, n), model.mask_id, device=device)
    is_masked = torch.ones((b, n), dtype=torch.bool, device=device)
    null_emb = model.null_cond.unsqueeze(0).repeat(b, 1)
    
    for step in range(12):
        both_tokens = torch.cat([tokens, tokens])
        both_cond = torch.cat([cond_emb, null_emb])
        logits = model(both_tokens, both_cond)
        c_logits, u_logits = logits.chunk(2)
        logits = u_logits + CONFIG["cfg_scale"] * (c_logits - u_logits)
        probs = F.softmax(logits / CONFIG["temp"], dim=-1)
        sampled = torch.distributions.Categorical(probs).sample()
        conf = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        conf[~is_masked] = float("inf")
        num_keep = max(1, int((1 - cosine_schedule((step + 1) / 12)) * n))
        _, idx = torch.topk(conf, num_keep, dim=-1)
        for i in range(b):
            tokens[i, idx[i]] = sampled[i, idx[i]]
            is_masked[i, idx[i]] = False

    indices = tokens.view(b, 32, 32)
    imgs = vqvae.decode_from_tokens(indices)
    Path(CONFIG["vis_folder"]).mkdir(exist_ok=True)
    vutils.save_image(imgs, f"{CONFIG['vis_folder']}/epoch_{epoch:03d}.png", nrow=b, normalize=False)
    torch.cuda.empty_cache()

def train():
    device = torch.device("cuda")
    model = MaskGIT(CONFIG["codebook_size"], CONFIG["seq_len"], CONFIG["clip_dim"], 
                    CONFIG["transformer_dim"], CONFIG["num_layers"], CONFIG["num_heads"]).to(device)
    
    vqvae = LFQ_VQVAE(embed_dim=CONFIG["embed_dim"]).to(device)
    vqvae.load_state_dict(torch.load(CONFIG["vqvae_weights"], map_location=device))
    vqvae.eval()
    
    pixelclip = PixelCLIP.load(CONFIG["pixelclip_weights"], device=device)
    pixelclip.eval()
    text_enc, _ = clip.load("ViT-B/32", device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
    loader = DataLoader(MaskGITDataset(CONFIG["cache_file"]), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    
    start_epoch = 1
    best_loss = float("inf")
    if Path(CONFIG["checkpoint_file"]).exists():
        ckpt = torch.load(CONFIG["checkpoint_file"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for tokens, clip_emb in pbar:
            tokens, clip_emb = tokens.to(device), clip_emb.to(device)
            b, n = tokens.shape
            drop = torch.rand(b, device=device) < 0.1
            clip_emb[drop] = model.null_cond
            
            mask_ratio = cosine_schedule(torch.rand(1).item())
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for i in range(b):
                perm = torch.randperm(n)[:max(1, int(mask_ratio * n))]
                mask[i, perm] = True
            
            input_tokens = tokens.clone()
            input_tokens[mask] = model.mask_id
            
            logits = model(input_tokens, clip_emb)
            
            # --- Chunked Loss to prevent OOM ---
            flat_logits = logits[mask]
            flat_targets = tokens[mask]
            chunk_size = 2048 # Lowered slightly for extra safety
            total_loss_val = 0
            
            optimizer.zero_grad()
            for j in range(0, flat_logits.size(0), chunk_size):
                end_j = min(j + chunk_size, flat_logits.size(0))
                l_chunk = F.cross_entropy(flat_logits[j:end_j], flat_targets[j:end_j], reduction='sum')
                l_chunk = l_chunk / flat_targets.size(0) # Normalize by total masked count
                l_chunk.backward(retain_graph=(j + chunk_size < flat_logits.size(0)))
                total_loss_val += l_chunk.item()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss_val
            pbar.set_postfix(loss=f"{total_loss_val:.4f}")
            del logits, flat_logits # Aggressive memory cleanup

        visualize(model, vqvae, pixelclip, text_enc, device, epoch)
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "epoch": epoch}, CONFIG["checkpoint_file"])

if __name__ == "__main__":
    train()
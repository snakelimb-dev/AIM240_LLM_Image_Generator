import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

# ── Config Block ───────────────────────────────
CONFIG = {
    "image_folder": r"C:\Users\abrax\Documents\2026 Python\MAgVitGBC\dataset_32x32",
    "checkpoint_file": "lfq_gan_vqvae_checkpoint.pt",
    "output_file": "12b_lfq_gan_vqvae_best.pt",
    "vis_folder": "visualizations4",
    "epochs": 150,
    "batch_size": 256,
    "lr": 3e-4,
    "embed_dim": 12,        
    "entropy_weight": 0.01,    # Lowered to prevent mode collapse
    "edge_weight": 0.0,        # Disabled temporarily to debug grey-out
    "adv_weight": 0.1,         # Weight of the GAN loss
    "disc_start_epoch": 5,    # Let the VAE warm up before the Discriminator kicks in
    "val_split": 0.005      
}
# ───────────────────────────────────────────────

class PixelDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((32, 32), Image.NEAREST)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return tensor

# ── Core Model Components ──────────────────────
class LookupFreeQuantization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        z_q = torch.sign(z)
        z_q = torch.where(z_q == 0, torch.ones_like(z_q), z_q)
        
        z_q_ste = z + (z_q - z).detach()
        entropy_loss = (z_q_ste.mean(dim=[0, 2, 3]) ** 2).mean()
        
        return z_q_ste, entropy_loss

    def quantize_to_indices(self, z_q):
        bits = (z_q > 0).int() 
        shifts = torch.arange(self.dim, device=z_q.device).view(1, -1, 1, 1)
        indices = (bits << shifts).sum(dim=1) 
        return indices

    def indices_to_quantized(self, indices):
        shifts = torch.arange(self.dim, device=indices.device).view(1, -1, 1, 1)
        indices = indices.unsqueeze(1) 
        bits = (indices >> shifts) & 1 
        z_q = bits.float() * 2.0 - 1.0 
        return z_q

class ResBlock(nn.Module):
    """Passes pixel data forward with GroupNorm to prevent grey mode-collapse."""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels)
        )

    def forward(self, x):
        return x + self.block(x)

class LFQ_VQVAE(nn.Module):
    def __init__(self, embed_dim=18):
        super().__init__()
        
        # Encoder: 1:1 spatial grid, stabilized with GroupNorm
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            ResBlock(256),
            
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1)
        )
        
        self.quantizer = LookupFreeQuantization(embed_dim)
        
        # Decoder: 1:1 spatial grid, stabilized with GroupNorm
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            ResBlock(256),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResBlock(128),
            
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, entropy_loss = self.quantizer(z)
        out = self.decoder(z_q)
        return out, entropy_loss

    @torch.no_grad()
    def encode_to_tokens(self, x):
        z = self.encoder(x)
        z_q, _ = self.quantizer(z)
        return self.quantizer.quantize_to_indices(z_q)

    @torch.no_grad()
    def decode_from_tokens(self, indices):
        z_q = self.quantizer.indices_to_quantized(indices)
        return self.decoder(z_q)

# ── PatchGAN Discriminator ─────────────────────
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ── Loss Functions ─────────────────────────────
def image_gradients(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy

def dual_reconstruction_loss(preds, targets, edge_weight):
    l1_loss = F.l1_loss(preds, targets)
    if edge_weight > 0.0:
        preds_dx, preds_dy = image_gradients(preds)
        targets_dx, targets_dy = image_gradients(targets)
        edge_loss = F.l1_loss(preds_dx, targets_dx) + F.l1_loss(preds_dy, targets_dy)
        return l1_loss + (edge_weight * edge_loss)
    return l1_loss

def d_hinge_loss(real_logits, fake_logits):
    loss_real = torch.mean(F.relu(1.0 - real_logits))
    loss_fake = torch.mean(F.relu(1.0 + fake_logits))
    return (loss_real + loss_fake) * 0.5

def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)

# ── Visualization ──────────────────────────────
@torch.no_grad()
def visualize_reconstructions(model, train_loader, val_loader, epoch, device):
    model.eval()
    train_imgs = next(iter(train_loader))[:2].to(device)
    val_imgs = next(iter(val_loader))[:2].to(device)
    
    all_imgs = torch.cat([train_imgs, val_imgs], dim=0)
    recons, _ = model(all_imgs)
    
    comparison = torch.empty(8, 3, 32, 32).to(device)
    comparison[0::2] = all_imgs
    comparison[1::2] = recons

    Path(CONFIG["vis_folder"]).mkdir(exist_ok=True)
    save_path = Path(CONFIG["vis_folder"]) / f"epoch_{epoch:03d}.png"
    vutils.save_image(comparison, save_path, nrow=2, padding=2, normalize=False)

# ── Training Loop ──────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    all_paths = [p for p in Path(CONFIG["image_folder"]).rglob("*") if p.suffix.lower() in exts]
    
    np.random.seed(42)
    np.random.shuffle(all_paths)
    
    val_size = max(1, int(len(all_paths) * CONFIG["val_split"]))
    train_paths = all_paths[:-val_size]
    val_paths = all_paths[-val_size:]
    print(f"Dataset split: {len(train_paths)} Train | {len(val_paths)} Val")

    train_loader = DataLoader(PixelDataset(train_paths), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(PixelDataset(val_paths), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Init Models & Optimizers
    vae = LFQ_VQVAE(CONFIG["embed_dim"]).to(device)
    disc = PatchDiscriminator().to(device)
    
    opt_vae = torch.optim.Adam(vae.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))

    start_epoch = 1
    best_loss = float("inf")

    # Resume Logic
    if Path(CONFIG["checkpoint_file"]).exists():
        print(f"Resuming from checkpoint...")
        ckpt = torch.load(CONFIG["checkpoint_file"], map_location=device)
        vae.load_state_dict(ckpt["vae_state"])
        disc.load_state_dict(ckpt["disc_state"])
        opt_vae.load_state_dict(ckpt["opt_vae_state"])
        opt_disc.load_state_dict(ckpt["opt_disc_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        vae.train()
        disc.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", unit="batch")
        for imgs in pbar:
            imgs = imgs.to(device)
            
            # 1. Forward VAE
            recons, entropy_loss = vae(imgs)
            
            use_adv = epoch >= CONFIG["disc_start_epoch"]
            adv_weight = CONFIG["adv_weight"] if use_adv else 0.0

            # 2. Train Discriminator
            d_loss_val = 0.0
            if use_adv:
                opt_disc.zero_grad()
                real_logits = disc(imgs)
                fake_logits = disc(recons.detach()) 
                
                d_loss = d_hinge_loss(real_logits, fake_logits)
                d_loss.backward()
                opt_disc.step()
                d_loss_val = d_loss.item()

            # 3. Train Generator (VAE)
            opt_vae.zero_grad()
            recon_loss = dual_reconstruction_loss(recons, imgs, CONFIG["edge_weight"])
            
            g_loss = recon_loss + (CONFIG["entropy_weight"] * entropy_loss)
            
            if use_adv:
                fake_logits = disc(recons) 
                adv_loss = g_hinge_loss(fake_logits)
                g_loss += adv_weight * adv_loss

            g_loss.backward()
            opt_vae.step()

            metrics = {"G_loss": f"{g_loss.item():.4f}"}
            if use_adv:
                metrics["D_loss"] = f"{d_loss_val:.4f}"
            pbar.set_postfix(**metrics)

        # Validation Phase
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                recons, entropy_loss = vae(imgs)
                recon_loss = dual_reconstruction_loss(recons, imgs, CONFIG["edge_weight"])
                val_loss += (recon_loss + (CONFIG["entropy_weight"] * entropy_loss)).item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"  ↳ Validation G_Loss: {avg_val_loss:.4f} {'(GAN Active)' if use_adv else '(Warmup)'}")

        visualize_reconstructions(vae, train_loader, val_loader, epoch, device)

        # Save Checkpoint
        torch.save({
            "epoch": epoch,
            "vae_state": vae.state_dict(),
            "disc_state": disc.state_dict(),
            "opt_vae_state": opt_vae.state_dict(),
            "opt_disc_state": opt_disc.state_dict(),
            "best_loss": best_loss
        }, CONFIG["checkpoint_file"])

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(vae.state_dict(), CONFIG["output_file"])
            print(f"  ★ New Best Val Loss: {best_loss:.4f} - Saved to {CONFIG['output_file']}")

    print("\nTraining complete.")

if __name__ == "__main__":
    train()
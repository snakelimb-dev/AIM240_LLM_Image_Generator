"""
PixelCLIP - Lightweight CLIP for color pixel art at 32x32.

Image encoder: small CNN, RGB input, native 32x32
Text encoder:  distilled from openai/clip-vit-base-patch32 (frozen, cached)

Usage:
    model = PixelCLIP()
    scores = model.rank(images, "a pixel art lion")
    best_idx = scores.argmax()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image


# ──────────────────────────────────────────────
# Tiny CNN image encoder — RGB in, 128-dim out
# ──────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    Accepts [B, 3, H, W] RGB, optimized for 32x32.
    Outputs [B, embed_dim] L2-normalized.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2 — 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3 — 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.MaxPool2d(2),

            # Block 4 — stays 4x4, deepens features
            nn.Conv2d(128, 256, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),           # 256*4*4 = 4096
            nn.Linear(4096, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


# ──────────────────────────────────────────────
# Text encoder — thin projection on CLIP text embeddings
# ──────────────────────────────────────────────

class TextProjection(nn.Module):
    def __init__(self, clip_dim=512, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


# ──────────────────────────────────────────────
# PixelCLIP
# ──────────────────────────────────────────────

class PixelCLIP(nn.Module):
    def __init__(self, embed_dim=128, device=None):
        super().__init__()
        self.embed_dim   = embed_dim
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder = ImageEncoder(embed_dim).to(self.device)
        self.text_proj     = TextProjection(512, embed_dim).to(self.device)
        self.logit_scale   = nn.Parameter(torch.tensor(2.659))
        self._clip_model   = None
        self._clip_proc    = None
        self._text_cache   = {}

    def _load_clip(self):
        if self._clip_model is not None:
            return
        try:
            import clip as openai_clip
            self._clip_model, self._clip_proc = openai_clip.load("ViT-B/32", device="cpu")
            self._clip_model.eval()
            print("[PixelCLIP] Loaded OpenAI CLIP ViT-B/32 for text encoding.")
        except ImportError:
            raise ImportError(
                "openai-clip not found.\n"
                "Install: pip install git+https://github.com/openai/CLIP.git"
            )

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        self._load_clip()
        import clip as openai_clip

        uncached = [p for p in prompts if p not in self._text_cache]
        if uncached:
            tokens = openai_clip.tokenize(uncached).to("cpu")
            raw    = self._clip_model.encode_text(tokens).float()
            raw    = F.normalize(raw, dim=-1)
            for p, r in zip(uncached, raw):
                self._text_cache[p] = r.cpu()

        stacked = torch.stack([self._text_cache[p] for p in prompts])
        return self.text_proj(stacked.to(self.device))

    def encode_text_raw(self, raw_embeds: torch.Tensor) -> torch.Tensor:
        return self.text_proj(raw_embeds.to(self.device))

    def preprocess_image(self, img) -> torch.Tensor:
        """
        Accepts: PIL Image, np.ndarray [H,W,3] or [H,W], torch tensor.
        Returns: [1, 3, H, W] float32 in [0, 1].
        Always uses NEAREST resampling.
        """
        if isinstance(img, torch.Tensor):
            t = img.float()
            if t.ndim == 2:                          # [H,W] grayscale -> RGB
                t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            elif t.ndim == 3 and t.shape[0] == 1:   # [1,H,W] -> RGB
                t = t.repeat(3, 1, 1).unsqueeze(0)
            elif t.ndim == 3 and t.shape[0] == 3:   # [3,H,W]
                t = t.unsqueeze(0)
            elif t.ndim == 4:
                pass
            return (t / 255.0 if t.max() > 1.5 else t).to(self.device)

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]                   # drop alpha
            img = Image.fromarray(
                img.astype(np.uint8) if img.max() > 1.5 else (img * 255).astype(np.uint8)
            )

        # PIL path
        img = img.convert("RGB")                     # handles L, RGBA, P, etc.
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def preprocess_batch(self, images: list) -> torch.Tensor:
        tensors = [self.preprocess_image(im) for im in images]
        sizes   = [(t.shape[2], t.shape[3]) for t in tensors]
        if len(set(sizes)) > 1:
            h = max(s[0] for s in sizes)
            w = max(s[1] for s in sizes)
            tensors = [F.interpolate(t, size=(h, w), mode="nearest") for t in tensors]
        return torch.cat(tensors, dim=0)

    def forward(self, images: torch.Tensor, text_embeds: torch.Tensor):
        img_emb = self.image_encoder(images)
        scale   = self.logit_scale.exp().clamp(1, 100)
        return scale * img_emb @ text_embeds.T

    @torch.no_grad()
    def rank(self, images: list, prompt: str) -> np.ndarray:
        batch   = self.preprocess_batch(images)
        txt_emb = self.encode_text([prompt])
        logits  = self.forward(batch, txt_emb)
        return logits[:, 0].cpu().numpy()

    @torch.no_grad()
    def best_match(self, images: list, prompt: str):
        scores = self.rank(images, prompt)
        return int(scores.argmax()), scores

    def save(self, path: str):
        torch.save({
            "image_encoder": self.image_encoder.state_dict(),
            "text_proj":     self.text_proj.state_dict(),
            "logit_scale":   self.logit_scale.data,
            "embed_dim":     self.embed_dim,
        }, path)
        print(f"[PixelCLIP] Saved to {path}")

    @classmethod
    def load(cls, path: str, device=None):
        ckpt  = torch.load(path, map_location="cpu")
        model = cls(embed_dim=ckpt["embed_dim"], device=device)
        model.image_encoder.load_state_dict(ckpt["image_encoder"])
        model.text_proj.load_state_dict(ckpt["text_proj"])
        model.logit_scale.data = ckpt["logit_scale"]
        print(f"[PixelCLIP] Loaded from {path}")
        return model


# ──────────────────────────────────────────────
# Distiller
# ──────────────────────────────────────────────

class PixelCLIPDistiller:
    """
    Distills from CLIP into PixelCLIP using a folder of pixel art images.
    No labels needed — matches CLIP's own image embeddings.

    Usage:
        distiller = PixelCLIPDistiller(model, "./pixel_art/")
        distiller.run(epochs=30)
        model.save("pixelclip.pt")
    """

    def __init__(self, model: PixelCLIP, image_folder: str, batch_size=64, lr=3e-4):
        self.model        = model
        self.image_folder = Path(image_folder)
        self.batch_size   = batch_size
        self.lr           = lr

    def _load_clip(self):
        self.model._load_clip()
        self._clip_model = self.model._clip_model
        self._clip_proc  = self.model._clip_proc

    @torch.no_grad()
    def _clip_image_embed(self, pil_img: Image.Image) -> torch.Tensor:
        """CLIP's own 512-dim image embedding — RGB 224x224 bilinear (CLIP's native pipeline)."""
        rgb    = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
        tensor = self._clip_proc(rgb).unsqueeze(0)
        emb    = self._clip_model.encode_image(tensor).float()
        return F.normalize(emb, dim=-1).squeeze(0)

    def run(self, epochs=30):
        self._load_clip()

        exts  = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        paths = [p for p in self.image_folder.rglob("*") if p.suffix.lower() in exts]
        if not paths:
            raise FileNotFoundError(f"No images found in {self.image_folder}")
        print(f"[Distiller] {len(paths)} images, {epochs} epochs")

        optimizer = torch.optim.AdamW(
            list(self.model.image_encoder.parameters()) +
            list(self.model.text_proj.parameters()),
            lr=self.lr, weight_decay=1e-4
        )
        device = self.model.device

        for epoch in range(1, epochs + 1):
            np.random.shuffle(paths)
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(paths), self.batch_size):
                batch_paths  = paths[i : i + self.batch_size]
                imgs_rgb     = []
                clip_targets = []

                for p in batch_paths:
                    try:
                        pil      = Image.open(p)
                        clip_emb = self._clip_image_embed(pil)
                        rgb      = self.model.preprocess_image(pil).squeeze(0)  # [3,H,W] native
                        imgs_rgb.append(rgb)
                        clip_targets.append(clip_emb)
                    except Exception as e:
                        print(f"  skip {p.name}: {e}")

                if len(imgs_rgb) < 2:
                    continue

                img_batch    = torch.stack(imgs_rgb).to(device)       # [B,3,H,W]
                clip_targets = torch.stack(clip_targets).to(device)   # [B,512]

                target_proj = self.model.encode_text_raw(clip_targets)  # [B,embed_dim]
                pred        = self.model.image_encoder(img_batch)       # [B,embed_dim]

                cos_loss = (1 - (pred * target_proj).sum(-1)).mean()

                scale   = self.model.logit_scale.exp().clamp(1, 100)
                logits  = scale * pred @ target_proj.T
                labels  = torch.arange(len(pred), device=device)
                nce     = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

                loss = cos_loss + nce

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if n_batches > 0:
                print(f"  Epoch {epoch}/{epochs}  loss={epoch_loss/n_batches:.4f}")


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== PixelCLIP (RGB) sanity check ===")
    model = PixelCLIP(embed_dim=128)
    print(f"Device: {model.device}")

    n_img = sum(p.numel() for p in model.image_encoder.parameters())
    n_txt = sum(p.numel() for p in model.text_proj.parameters())
    print(f"Image encoder params: {n_img:,}")
    print(f"Text proj params:     {n_txt:,}")
    print(f"Total trainable:      {n_img + n_txt:,}")

    rng = np.random.default_rng(42)

    # Test RGB numpy arrays [H,W,3]
    fake_rgb = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
    try:
        scores = model.rank(fake_rgb, "a pixel art lion")
        print(f"\nRGB 32x32 scores: {scores}")
        best, _ = model.best_match(fake_rgb, "a pixel art lion")
        print(f"Best match idx: {best}")
    except Exception as e:
        print(f"RGB test FAILED: {e}")
        sys.exit(1)

    # Test PIL RGBA (common for pixel art PNGs with transparency)
    fake_rgba = [Image.fromarray(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8), mode="RGBA")
                 for _ in range(3)]
    try:
        scores = model.rank(fake_rgba, "a pixel art sword")
        print(f"RGBA 32x32 scores: {scores}")
    except Exception as e:
        print(f"RGBA test FAILED: {e}")
        sys.exit(1)

    print("\nAll checks passed. (Scores are random — model not yet trained.)")
    print("\nTo train:")
    print("  python train_pixelclip.py --images ./your_pixel_art --output pixelclip.pt")

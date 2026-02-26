import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from PIL import Image

from pixelclip import PixelCLIP

# ── Config ─────────────────────────────────────
IMAGE_FOLDER  = r"C:\Users\abrax\Documents\2026 Python\MAgVitGBC\dataset_32x32"  # change this
OUTPUT_FILE   = "pixelclip.pt"
CHECKPOINT    = "pixelclip_checkpoint.pt"   # resumes from here if it exists
EPOCHS        = 30
BATCH_SIZE    = 256
LR            = 3e-4
EMBED_DIM     = 512
# ───────────────────────────────────────────────


def load_clip():
    try:
        import clip as openai_clip
        model, proc = openai_clip.load("ViT-B/32", device="cpu")
        model.eval()
        print("[CLIP] Loaded ViT-B/32")
        return model, proc
    except ImportError:
        raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")


@torch.no_grad()
def clip_image_embed(pil_img, clip_model, clip_proc):
    rgb    = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
    tensor = clip_proc(rgb).unsqueeze(0)
    emb    = clip_model.encode_image(tensor).float()
    return F.normalize(emb, dim=-1).squeeze(0)


def train():
    folder = Path(IMAGE_FOLDER)
    if not folder.exists():
        print(f"ERROR: image folder not found: {folder}")
        return

    exts  = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        print(f"ERROR: no images found in {folder}")
        return
    print(f"Found {len(paths)} images")

    # ── Model ──
    model = PixelCLIP(embed_dim=EMBED_DIM)
    device = model.device

    optimizer = torch.optim.AdamW(
        list(model.image_encoder.parameters()) +
        list(model.text_proj.parameters()),
        lr=LR, weight_decay=1e-4
    )

    start_epoch = 1
    best_loss   = float("inf")
    loss_history = []

    # ── Resume ──
    if Path(CHECKPOINT).exists():
        print(f"Resuming from {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location="cpu")
        model.image_encoder.load_state_dict(ckpt["image_encoder"])
        model.text_proj.load_state_dict(ckpt["text_proj"])
        model.logit_scale.data = ckpt["logit_scale"]
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_loss    = ckpt.get("best_loss", float("inf"))
        loss_history = ckpt.get("loss_history", [])
        print(f"  Resumed at epoch {start_epoch}, best loss so far: {best_loss:.4f}")
    else:
        print("No checkpoint found, starting fresh")

    # ── CLIP ──
    print("Loading CLIP teacher...")
    clip_model, clip_proc = load_clip()

    print(f"\n{'='*55}")
    print(f"  Epochs:     {start_epoch} -> {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR:         {LR}")
    print(f"  Device:     {device}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params:     {n_params:,}")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        np.random.shuffle(paths)

        epoch_loss   = 0.0
        epoch_cos    = 0.0
        epoch_nce    = 0.0
        n_batches    = 0
        n_skipped    = 0

        total_batches = max(1, len(paths) // BATCH_SIZE)

        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths  = paths[i : i + BATCH_SIZE]
            imgs_rgb     = []
            clip_targets = []

            for p in batch_paths:
                try:
                    pil      = Image.open(p)
                    clip_emb = clip_image_embed(pil, clip_model, clip_proc)
                    rgb      = model.preprocess_image(pil).squeeze(0)
                    imgs_rgb.append(rgb)
                    clip_targets.append(clip_emb)
                except Exception as e:
                    n_skipped += 1

            if len(imgs_rgb) < 2:
                continue

            img_batch    = torch.stack(imgs_rgb).to(device)
            clip_targets = torch.stack(clip_targets).to(device)

            target_proj = model.encode_text_raw(clip_targets)
            pred        = model.image_encoder(img_batch)

            cos_loss = (1 - (pred * target_proj).sum(-1)).mean()

            scale   = model.logit_scale.exp().clamp(1, 100)
            logits  = scale * pred @ target_proj.T
            labels  = torch.arange(len(pred), device=device)
            nce     = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            loss = cos_loss + nce

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos  += cos_loss.item()
            epoch_nce  += nce.item()
            n_batches  += 1

            # ── Batch progress ──
            batch_num = n_batches
            pct       = 100 * i / len(paths)
            print(f"  Epoch {epoch}/{EPOCHS}  "
                  f"batch {batch_num}/{total_batches}  "
                  f"({pct:.0f}%)  "
                  f"loss={loss.item():.4f}  "
                  f"cos={cos_loss.item():.4f}  "
                  f"nce={nce.item():.4f}",
                  flush=True)

        if n_batches == 0:
            print(f"Epoch {epoch}: no valid batches, skipping")
            continue

        avg_loss = epoch_loss / n_batches
        avg_cos  = epoch_cos  / n_batches
        avg_nce  = epoch_nce  / n_batches
        elapsed  = time.time() - epoch_start
        loss_history.append(avg_loss)

        print(f"\n── Epoch {epoch}/{EPOCHS} complete ──────────────────────")
        print(f"   avg loss : {avg_loss:.4f}  (cos={avg_cos:.4f}  nce={avg_nce:.4f})")
        print(f"   skipped  : {n_skipped} images")
        print(f"   time     : {elapsed:.1f}s")
        if len(loss_history) >= 2:
            delta = loss_history[-1] - loss_history[-2]
            arrow = "▼" if delta < 0 else "▲"
            print(f"   change   : {arrow} {abs(delta):.4f}")
        print(f"─────────────────────────────────────────────────────\n")

        # ── Checkpoint every epoch ──
        torch.save({
            "image_encoder": model.image_encoder.state_dict(),
            "text_proj":     model.text_proj.state_dict(),
            "logit_scale":   model.logit_scale.data,
            "optimizer":     optimizer.state_dict(),
            "epoch":         epoch,
            "best_loss":     best_loss,
            "loss_history":  loss_history,
            "embed_dim":     EMBED_DIM,
        }, CHECKPOINT)

        # ── Save best ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(OUTPUT_FILE)
            print(f"   ★ New best! Saved to {OUTPUT_FILE}\n")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {OUTPUT_FILE}")
    print(f"\nLoss history:")
    for i, l in enumerate(loss_history, 1):
        bar = "█" * int(20 * (1 - l / max(loss_history)))
        print(f"  Epoch {i:>3}: {l:.4f}  {bar}")


if __name__ == "__main__":
    train()
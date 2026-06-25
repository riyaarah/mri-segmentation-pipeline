"""
Model architecture and inference logic for MRI brain tumor segmentation.
Shared by both the FastAPI service and the Streamlit demo.
"""
import os
import numpy as np
import torch
import torch.nn as nn


# ── Model Definition ──────────────────────────────────────────
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3D(ch, f))
            self.pools.append(nn.MaxPool3d(2))
            ch = f
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConv3D(f * 2, f))
            ch = f
        self.final = nn.Conv3d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final(x)


# ── Preprocessing helpers ─────────────────────────────────────
def normalize(vol):
    mi, ma = vol.min(), vol.max()
    return (vol - mi) / (ma - mi) if ma != mi else vol


def crop_center(vol, size=(96, 96, 96)):
    x, y, z = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    sx, sy, sz = (x - size[0]) // 2, (y - size[1]) // 2, (z - size[2]) // 2
    if vol.ndim == 4:
        return vol[:, sx:sx + size[0], sy:sy + size[1], sz:sz + size[2]]
    return vol[sx:sx + size[0], sy:sy + size[1], sz:sz + size[2]]


# ── Model loading (singleton pattern — load once, reuse) ──────
_model = None
_device = None

# Local dev: looks for the checkpoint in models/. Deployed (HF Spaces):
# falls back to downloading it from the HF Model repo at startup, since
# the checkpoint isn't shipped inside the Docker image.
MODEL_PATH = os.environ.get("MRI_MODEL_PATH", "models/best_model_colab.pth")
HF_MODEL_REPO = os.environ.get("MRI_HF_MODEL_REPO", "riyaarahim/mri-tumor-segmentation-unet")
HF_MODEL_FILENAME = os.environ.get("MRI_HF_MODEL_FILENAME", "best_model_colab.pth")


def _resolve_model_path() -> str:
    """Return a local path to the checkpoint, downloading from the Hub if needed."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    # Not found locally — download from the HF Model repo (cached after first run)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)


def get_model():
    """Lazily load and cache the model. Call this from API startup."""
    global _model, _device
    if _model is None:
        _device = torch.device("cpu")
        _model = UNet3D(in_channels=4, num_classes=4, features=[32, 64, 128, 256])
        resolved_path = _resolve_model_path()
        _model.load_state_dict(torch.load(resolved_path, map_location=_device))
        _model.eval()
    return _model, _device


# ── Inference ──────────────────────────────────────────────────
def predict(flair: np.ndarray, t1: np.ndarray, t1ce: np.ndarray, t2: np.ndarray):
    """
    Run segmentation on 4 raw MRI volumes (already loaded as numpy arrays).
    Returns: (cropped_input_volume, predicted_mask) both as numpy arrays.
    """
    model, device = get_model()

    combined = np.stack([
        normalize(flair), normalize(t1), normalize(t1ce), normalize(t2)
    ], axis=0)
    cropped = crop_center(combined)

    x = torch.tensor(cropped[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x)
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    return cropped, mask


def compute_stats(mask: np.ndarray) -> dict:
    """Compute tumor region statistics from a predicted mask."""
    total = mask.size
    return {
        "necrotic_core_pct": round(float((mask == 1).sum()) / total * 100, 2),
        "edema_pct": round(float((mask == 2).sum()) / total * 100, 2),
        "enhancing_tumor_pct": round(float((mask == 3).sum()) / total * 100, 2),
        "total_tumor_pct": round(float((mask > 0).sum()) / total * 100, 2),
    }

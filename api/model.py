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


# ── Result caching (so /predict/slice can render slices on demand
# without re-running inference, and the frontend doesn't need to
# transmit raw numpy arrays over HTTP) ─────────────────────────
import time
import uuid

_RESULT_CACHE: dict[str, dict] = {}
_CACHE_TTL_SECONDS = 30 * 60  # evict results after 30 minutes


def cache_result(volume: np.ndarray, mask: np.ndarray) -> str:
    """Store a prediction result server-side, return a job_id to retrieve it later."""
    _evict_expired()
    job_id = str(uuid.uuid4())
    _RESULT_CACHE[job_id] = {
        "volume": volume,
        "mask": mask,
        "created_at": time.time(),
    }
    return job_id


def get_cached_result(job_id: str) -> dict | None:
    _evict_expired()
    return _RESULT_CACHE.get(job_id)


def _evict_expired():
    now = time.time()
    expired = [k for k, v in _RESULT_CACHE.items() if now - v["created_at"] > _CACHE_TTL_SECONDS]
    for k in expired:
        del _RESULT_CACHE[k]


# ── Slice rendering (API owns visualization, frontend just displays) ──
def render_slice_image(volume: np.ndarray, mask: np.ndarray, slice_idx: int) -> bytes:
    """
    Render the 3-panel slice view (FLAIR input / predicted mask / region
    overlay) for a single slice index, returned as PNG bytes.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")  # headless rendering, no display needed on the server
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    flair_slice = volume[0, :, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title('FLAIR Input', color='white', fontsize=13)
    axes[0].axis('off')

    cmap = plt.cm.get_cmap('jet', 4)
    axes[1].imshow(flair_slice, cmap='gray')
    axes[1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                   cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title('Predicted Tumor Mask', color='white', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(flair_slice, cmap='gray')
    axes[2].imshow(np.ma.masked_where(mask_slice != 1, mask_slice),
                   cmap='Reds', alpha=0.7, vmin=0, vmax=3)
    axes[2].imshow(np.ma.masked_where(mask_slice != 2, mask_slice),
                   cmap='Greens', alpha=0.7, vmin=0, vmax=3)
    axes[2].imshow(np.ma.masked_where(mask_slice != 3, mask_slice),
                   cmap='Blues', alpha=0.7, vmin=0, vmax=3)
    axes[2].set_title('Region Overlay', color='white', fontsize=13)
    axes[2].axis('off')

    patches = [
        mpatches.Patch(color='red',   label='Necrotic Core'),
        mpatches.Patch(color='green', label='Edema'),
        mpatches.Patch(color='blue',  label='Enhancing Tumor'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               facecolor='#0e1117', labelcolor='white', fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
